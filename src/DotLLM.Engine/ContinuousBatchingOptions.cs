namespace DotLLM.Engine;

/// <summary>Limits for iteration-level batching with chunked prefill.</summary>
public sealed record ContinuousBatchingOptions
{
    /// <summary>Maximum concurrently resident sequences.</summary>
    public int MaxSequences { get; init; } = 8;
    /// <summary>Total input tokens per forward. Must exceed MaxSequences to allow prefill progress.</summary>
    public int MaxBatchedTokens { get; init; } = 256;
    /// <summary>Maximum prompt tokens from one sequence per iteration.</summary>
    public int PrefillChunkSize { get; init; } = 128;
    /// <summary>Maximum requests waiting for admission.</summary>
    public int MaxQueuedRequests { get; init; } = 128;
    /// <summary>Buffered tokens per stream. Slow consumers fail without blocking other requests.</summary>
    public int StreamBufferCapacity { get; init; } = 64;

    /// <summary>Rejects invalid or non-progressing scheduler limits.</summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxSequences);
        if (MaxBatchedTokens <= MaxSequences)
            throw new ArgumentException("MaxBatchedTokens must exceed MaxSequences.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PrefillChunkSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxQueuedRequests);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(StreamBufferCapacity);
    }
}
