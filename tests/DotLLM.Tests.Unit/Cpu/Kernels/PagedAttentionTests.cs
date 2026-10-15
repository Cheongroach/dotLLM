using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class PagedAttentionTests
{
    [Theory]
    [InlineData(4, 4, 16, 19, 0, 0, false)]
    [InlineData(4, 2, 32, 273, 265, 0, false)]
    [InlineData(4, 1, 64, 273, 272, 0, true)]
    [InlineData(4, 2, 32, 273, 253, 17, true)]
    public void FragmentedPagesMatchScalar(int heads, int kvHeads, int dim,
        int length, int offset, int window, bool parallel)
    {
        const int blockSize = 16;
        int stride = kvHeads * dim;
        using var pool = new KvBlockPool(2, kvHeads, dim, blockSize, totalBlocks: 64);
        using var cache = new PagedKvCache(pool, 2, stride, length);
        var blockers = new List<int>();
        var rng = new Random(42);
        float[] keys = RandomValues(rng, length * stride);
        float[] values = RandomValues(rng, length * stride);
        // Interleave this sequence's pages with live pages belonging to other sequences.
        fixed (float* kp = keys, vp = values)
        {
            for (int pos = 0; pos < length; pos += blockSize)
            {
                blockers.Add(pool.Allocate());
                int count = Math.Min(blockSize, length - pos);
                int[] positions = Enumerable.Range(pos, count).ToArray();
                cache.Update(new TensorRef(count, stride, DType.Float32, -1, (nint)(kp + pos * stride)),
                    new TensorRef(count, stride, DType.Float32, -1, (nint)(vp + pos * stride)), positions, 1);
            }
        }
        int queries = length - offset;
        float[] q = RandomValues(rng, queries * heads * dim);
        float[] expected = new float[q.Length];
        float[] actual = new float[q.Length];
        int? sw = window > 0 ? window : null;
        Attention.ExecuteScalar(q, keys, values, expected, queries, length, heads, kvHeads, dim, offset, sw);
        using var workers = parallel ? new ComputeThreadPool(2) : null;
        fixed (float* qp = q, op = actual)
            PagedAttention.Execute(qp, cache, 1, op, queries, heads, kvHeads, dim, offset, workers, sw);
        for (int i = 0; i < actual.Length; i++)
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0, 0.001f);
        Assert.Equal(0, cache.StagingBytes);
        foreach (int id in blockers) pool.Release(id);
        cache.Dispose();
        Assert.Equal(pool.TotalBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void RollbackAndOverwriteUseLogicalPages()
    {
        using var pool = new KvBlockPool(1, 1, 4, 2, 8);
        using var cache = new PagedKvCache(pool, 1, 4, 8);
        float[] data = Enumerable.Range(1, 16).Select(i => (float)i).ToArray();
        fixed (float* p = data)
        {
            cache.Update(new TensorRef(4, 4, DType.Float32, -1, (nint)p),
                new TensorRef(4, 4, DType.Float32, -1, (nint)p), [0, 1, 2, 3], 0);
            cache.Rollback(1);
            cache.Update(new TensorRef(1, 4, DType.Float32, -1, (nint)p),
                new TensorRef(1, 4, DType.Float32, -1, (nint)p), [1], 0);
            float* output = stackalloc float[4];
            PagedAttention.Execute(p, cache, 0, output, 1, 1, 1, 4, 1);
            for (int i = 0; i < 4; i++) Assert.Equal(data[i], output[i], 5);
        }
        Assert.Equal(0, cache.StagingBytes);
    }

    private static float[] RandomValues(Random rng, int count)
        => Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
}
