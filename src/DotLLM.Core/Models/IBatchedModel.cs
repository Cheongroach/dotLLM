using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Models;

/// <summary>
/// Shares projection work across independent sequences in a packed token batch.
/// One model instance has a single compute owner; concurrent forwards are not supported.
/// </summary>
public interface IBatchedModel : IModel
{
    /// <summary>
    /// Runs unpadded sequence segments with isolated attention/cache updates. Returns owned
    /// CPU Float32 logits [sequences.Length, vocabSize], one last-token row per segment.
    /// </summary>
    ITensor ForwardBatch(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
        ReadOnlySpan<SequenceBatch> sequences, int deviceId);
}

/// <summary>A contiguous segment of a packed batch and its exclusively owned cache.</summary>
/// <param name="TokenOffset">Offset into packed token/position buffers.</param>
/// <param name="TokenCount">Number of tokens to append.</param>
/// <param name="Cache">Cache containing this sequence's computed prefix.</param>
public readonly record struct SequenceBatch(int TokenOffset, int TokenCount, IKvCache Cache)
{
    /// <summary>Validates an append-only batch before any cache or model state is mutated.</summary>
    public static void Validate(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions,
        ReadOnlySpan<SequenceBatch> sequences, ModelConfig config)
    {
        if (tokens.IsEmpty || tokens.Length != positions.Length || sequences.IsEmpty)
            throw new ArgumentException("A batch needs tokens, matching positions, and sequence segments.");

        int offset = 0;
        for (int s = 0; s < sequences.Length; s++)
        {
            var sequence = sequences[s];
            ArgumentNullException.ThrowIfNull(sequence.Cache);
            if (sequence.TokenOffset != offset || sequence.TokenCount <= 0 ||
                sequence.TokenCount > tokens.Length - offset)
                throw new ArgumentException("Segments must partition the batch.", nameof(sequences));
            for (int prior = 0; prior < s; prior++)
                if (ReferenceEquals(sequences[prior].Cache, sequence.Cache))
                    throw new ArgumentException("Sequences must own distinct caches.", nameof(sequences));

            int start = sequence.Cache.CurrentLength;
            for (int t = 0; t < sequence.TokenCount; t++)
            {
                int index = offset + t;
                if ((uint)tokens[index] >= (uint)config.VocabSize ||
                    (uint)positions[index] >= (uint)Math.Min(config.MaxSequenceLength, sequence.Cache.MaxLength) ||
                    positions[index] != (long)start + t)
                    throw new ArgumentException("Invalid token or non-contiguous append.", nameof(positions));
            }
            offset += sequence.TokenCount;
        }
        if (offset != tokens.Length)
            throw new ArgumentException("Segments must cover every token.", nameof(sequences));
    }
}
