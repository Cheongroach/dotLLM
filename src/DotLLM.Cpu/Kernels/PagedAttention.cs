using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Float32 attention over physical KV pages. Uses bounded score tiles and online softmax;
/// neither a contiguous KV copy nor a full sequence-sized score matrix is materialized.
/// </summary>
public static unsafe class PagedAttention
{
    private const int TileSize = 256;

    /// <summary>
    /// Computes attention for contiguous query positions starting at <paramref name="positionOffset"/>.
    /// Supports MHA/MQA/GQA, causal masking, sliding windows, and partial/non-contiguous pages.
    /// Input and output layouts are [seqQ, numHeads * headDim], on the CPU.
    /// </summary>
    public static void Execute(float* q, IPagedKvCache cache, int layerIndex, float* output,
        int seqQ, int numHeads, int numKvHeads, int headDim, int positionOffset,
        ComputeThreadPool? pool = null, int? slidingWindowSize = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numKvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqQ);
        if (numHeads % numKvHeads != 0 || cache.KvStride != checked(numKvHeads * headDim))
            throw new ArgumentException("Attention heads/stride must match the cache.");
        if (cache.BlockSize <= 0 || positionOffset < 0 ||
            (long)positionOffset + seqQ > cache.CurrentLength)
            throw new ArgumentException("Query positions must be present in the cache.");
        if (slidingWindowSize is <= 0)
            throw new ArgumentOutOfRangeException(nameof(slidingWindowSize));

        var keys = cache.GetKeyBlocksRef(layerIndex);
        var values = cache.GetValueBlocksRef(layerIndex);
        if (keys.DType != DType.Float32 || values.DType != DType.Float32 ||
            keys.DeviceId != -1 || values.DeviceId != -1)
            throw new ArgumentException("PagedAttention requires CPU Float32 pages.");
        int usedBlocks = (cache.CurrentLength - 1) / cache.BlockSize + 1;
        var blocks = cache.BlockIds;
        if (blocks.Length < usedBlocks)
            throw new ArgumentException("Incomplete block table.", nameof(cache));
        for (int i = 0; i < usedBlocks; i++)
            if ((uint)blocks[i] >= (uint)(keys.Dim0 / cache.BlockSize) ||
                (uint)blocks[i] >= (uint)(values.Dim0 / cache.BlockSize))
                throw new ArgumentException("Block ID outside physical storage.", nameof(cache));

        fixed (int* table = blocks)
        {
            var context = new Context
            {
                Q = q, K = (float*)keys.DataPointer, V = (float*)values.DataPointer,
                Output = output, Blocks = table, BlockSize = cache.BlockSize,
                SeqQ = seqQ, NumHeads = numHeads, GroupSize = numHeads / numKvHeads,
                HeadDim = headDim, KvStride = cache.KvStride, QStride = checked(numHeads * headDim),
                PositionOffset = positionOffset, Window = slidingWindowSize ?? 0,
                Scale = 1f / MathF.Sqrt(headDim)
            };
            if (pool is null)
                Worker((nint)(&context), 0, 1);
            else
                pool.Dispatch((nint)(&context), &Worker);
        }
    }

    private struct Context
    {
        public float* Q;
        public float* K;
        public float* V;
        public float* Output;
        public int* Blocks;
        public int BlockSize, SeqQ, NumHeads, GroupSize, HeadDim, KvStride, QStride;
        public int PositionOffset, Window;
        public float Scale;
    }

    [SkipLocalsInit]
    private static void Worker(nint contextPtr, int workerIndex, int workerCount)
    {
        ref var ctx = ref Unsafe.AsRef<Context>((void*)contextPtr);
        Span<float> tileScores = stackalloc float[TileSize];
        // Heads stay on the same worker across queries, improving KV locality.
        for (int h = workerIndex; h < ctx.NumHeads; h += workerCount)
        {
            int kvHeadOffset = h / ctx.GroupSize * ctx.HeadDim;
            for (int t = 0; t < ctx.SeqQ; t++)
            {
                var query = new ReadOnlySpan<float>(ctx.Q + (long)t * ctx.QStride + h * ctx.HeadDim, ctx.HeadDim);
                var result = new Span<float>(ctx.Output + (long)t * ctx.QStride + h * ctx.HeadDim, ctx.HeadDim);
                result.Clear();
                int end = ctx.PositionOffset + t + 1;
                int start = ctx.Window > 0 ? Math.Max(0, end - ctx.Window) : 0;
                float max = float.NegativeInfinity;
                float sum = 0f;

                for (int tile = start; tile < end; tile += TileSize)
                {
                    int count = Math.Min(TileSize, end - tile);
                    var scores = tileScores[..count];
                    for (int j = 0; j < count; j++)
                    {
                        long offset = Resolve(ref ctx, tile + j) + kvHeadOffset;
                        scores[j] = TensorPrimitives.Dot(query,
                            new ReadOnlySpan<float>(ctx.K + offset, ctx.HeadDim)) * ctx.Scale;
                    }
                    float nextMax = MathF.Max(max, TensorPrimitives.Max(scores));
                    float correction = MathF.Exp(max - nextMax);
                    sum *= correction;
                    TensorPrimitives.Multiply(result, correction, result);
                    TensorPrimitives.Subtract(scores, nextMax, scores);
                    TensorPrimitives.Exp(scores, scores);
                    sum += TensorPrimitives.Sum(scores);

                    for (int j = 0; j < count; j++)
                    {
                        long offset = Resolve(ref ctx, tile + j) + kvHeadOffset;
                        TensorPrimitives.MultiplyAdd(
                            new ReadOnlySpan<float>(ctx.V + offset, ctx.HeadDim), scores[j], result, result);
                    }
                    max = nextMax;
                }
                if (sum > 0)
                    TensorPrimitives.Multiply(result, 1f / sum, result);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long Resolve(ref Context ctx, int position)
        => ((long)ctx.Blocks[position / ctx.BlockSize] * ctx.BlockSize + position % ctx.BlockSize) * ctx.KvStride;
}
