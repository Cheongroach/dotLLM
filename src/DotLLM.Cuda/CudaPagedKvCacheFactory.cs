using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Fixed-capacity FP16 GPU page pool shared by independent sequences. Uses one block ID
/// across all layers. Single compute owner; dispose caches before the pool, and the pool before its context.
/// </summary>
public sealed class CudaPagedKvCacheFactory : IDisposable
{
    private readonly CudaContext _context;
    private readonly Stack<int> _free;
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private bool _disposed;
    internal int KvStride { get; }
    internal long BytesPerBlock { get; }
    internal nint ContextHandle => _context.Handle;
    internal int NumLayers => _keys.Length;
    /// <summary>Number of tokens per physical page.</summary>
    public int BlockSize { get; }
    /// <summary>Total physical pages available to the scheduler.</summary>
    public int TotalBlocks { get; }
    /// <summary>Number of currently unused pages.</summary>
    public int FreeBlocks => _free.Count;
    /// <summary>Fixed GPU KV storage size, excluding per-sequence page tables.</summary>
    public long AllocatedBytes => TotalBlocks * BytesPerBlock;

    /// <summary>Allocates a pool on the supplied context. The caller retains ownership of the context.</summary>
    public CudaPagedKvCacheFactory(CudaContext context, int numLayers, int numKvHeads, int headDim,
        int blockSize = 16, int maxTotalTokens = 65536)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numKvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTotalTokens);
        _context = context;
        BlockSize = blockSize;
        KvStride = checked(numKvHeads * headDim);
        TotalBlocks = checked((int)(((long)maxTotalTokens + blockSize - 1) / blockSize));
        BytesPerBlock = checked(2L * numLayers * blockSize * KvStride * sizeof(ushort));
        _keys = new nint[numLayers];
        _values = new nint[numLayers];
        _free = new Stack<int>(TotalBlocks);
        for (int i = TotalBlocks - 1; i >= 0; i--) _free.Push(i);
        MakeCurrent();
        try
        {
            nuint bytes = checked((nuint)((long)TotalBlocks * blockSize * KvStride * sizeof(ushort)));
            for (int i = 0; i < numLayers; i++)
            {
                CudaDriverApi.cuMemAlloc_v2(out _keys[i], bytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out _values[i], bytes).ThrowOnError();
            }
        }
        catch { Dispose(); throw; }
    }

    /// <summary>Creates an empty request cache. Pages are allocated as the sequence grows.</summary>
    public CudaPagedKvCache Create(int maxSeqLen)
    {
        MakeCurrent();
        return new CudaPagedKvCache(this, maxSeqLen);
    }
    internal void MakeCurrent()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _context.MakeCurrent();
    }
    internal int Rent() => _free.Pop();
    internal void Return(int block) => _free.Push(block);
    internal nint Keys(int layer) => _keys[layer];
    internal nint Values(int layer) => _values[layer];

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_free.Count != TotalBlocks) throw new InvalidOperationException("Dispose sequence caches before their GPU page pool.");
        MakeCurrent();
        for (int i = 0; i < _keys.Length; i++)
        {
            if (_keys[i] != 0) { CudaDriverApi.cuMemFree_v2(_keys[i]); _keys[i] = 0; }
            if (_values[i] != 0) { CudaDriverApi.cuMemFree_v2(_values[i]); _values[i] = 0; }
        }
        _disposed = true;
    }
}
