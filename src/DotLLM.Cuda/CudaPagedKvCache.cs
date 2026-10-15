using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// An append-only sequence's GPU page table. The factory owns storage; disposing a cache
/// returns its pages. Access is serialized by the model/scheduler compute owner.
/// </summary>
public sealed unsafe class CudaPagedKvCache : IKvCache
{
    internal CudaPagedKvCacheFactory Factory { get; }
    private readonly int[] _blocks;
    private int _blockCount;
    private nint _tableDevice;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength { get; private set; }
    /// <inheritdoc/>
    public int MaxLength { get; }
    /// <summary>Bytes occupied by this sequence's physical KV pages (excluding its page table).</summary>
    public long AllocatedBytes => (long)_blockCount * Factory.BytesPerBlock;
    internal nint TableDevice => _tableDevice;

    internal CudaPagedKvCache(CudaPagedKvCacheFactory factory, int maxLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        Factory = factory;
        MaxLength = maxLength;
        _blocks = new int[checked((int)(((long)maxLength + factory.BlockSize - 1) / factory.BlockSize))];
        factory.MakeCurrent();
        CudaDriverApi.cuMemAlloc_v2(out _tableDevice, (nuint)((long)_blocks.Length * sizeof(int))).ThrowOnError();
    }

    // Called once before launching a forward. Previous forwards must have synchronized.
    internal void Prepare(int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (length < CurrentLength || length > MaxLength) throw new ArgumentOutOfRangeException(nameof(length));
        Factory.MakeCurrent();
        int required = (int)(((long)length + Factory.BlockSize - 1) / Factory.BlockSize);
        int oldCount = _blockCount;
        if (required - oldCount > Factory.FreeBlocks)
            throw new InvalidOperationException("GPU KV page pool exhausted.");
        while (_blockCount < required) _blocks[_blockCount++] = Factory.Rent();
        if (_blockCount != oldCount)
        {
            fixed (int* blocks = _blocks)
                CudaDriverApi.cuMemcpyHtoD_v2(_tableDevice, (nint)blocks,
                    (nuint)((long)_blockCount * sizeof(int))).ThrowOnError();
        }
        CurrentLength = length;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)length > (uint)CurrentLength) throw new ArgumentOutOfRangeException(nameof(length));
        int keep = (int)(((long)length + Factory.BlockSize - 1) / Factory.BlockSize);
        while (_blockCount > keep) Factory.Return(_blocks[--_blockCount]);
        CurrentLength = length;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use the CUDA model's device page update.");
    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use the CUDA model's device page update.");
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("GPU KV pages are not contiguous.");
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => throw new NotSupportedException("GPU KV pages are not contiguous.");
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => throw new NotSupportedException("GPU KV pages are not contiguous.");
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => throw new NotSupportedException("GPU KV pages are not contiguous.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        Factory.MakeCurrent();
        Rollback(0);
        CudaDriverApi.cuMemFree_v2(_tableDevice).ThrowOnError();
        _tableDevice = 0;
        _disposed = true;
    }
}
