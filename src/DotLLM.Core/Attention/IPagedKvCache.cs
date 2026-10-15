using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// CPU Float32 paged storage. Kernels read physical blocks directly without gathering.
/// Views are borrowed until the next cache mutation or disposal.
/// </summary>
public interface IPagedKvCache : IKvCache
{
    /// <summary>Tokens in a physical block.</summary>
    int BlockSize { get; }
    /// <summary>Elements per token (KV heads times head dimension).</summary>
    int KvStride { get; }
    /// <summary>Physical block IDs in logical token order.</summary>
    ReadOnlySpan<int> BlockIds { get; }
    /// <summary>Base of all physical key blocks for one layer.</summary>
    TensorRef GetKeyBlocksRef(int layerIndex);
    /// <summary>Base of all physical value blocks for one layer.</summary>
    TensorRef GetValueBlocksRef(int layerIndex);
}
