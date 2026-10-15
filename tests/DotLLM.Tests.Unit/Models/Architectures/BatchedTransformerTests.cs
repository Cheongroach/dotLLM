using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

public sealed class BatchedTransformerTests
{
    [Theory]
    [InlineData(1, 0, 1)]
    [InlineData(2, 0, 2)]
    [InlineData(4, 0, 1)]
    [InlineData(2, 5, 2)]
    public unsafe void PackedLogitsMatchIndependentSequences(int kvHeads, int window, int threads)
    {
        using var fixture = new TinyTransformer(kvHeads, window, threads);
        var config = fixture.Model.Config;
        using var pool = new PagedKvCacheFactory(2, kvHeads, 8, blockSize: 2, maxTotalTokens: 128);
        using var a = pool.Create(64);
        using var b = pool.Create(64);
        // Keep the attention kernel identical: legacy contiguous attention uses approximate exp.
        using var serialA = pool.Create(64);
        using var serialB = pool.Create(64);
        int[] aTokens = [1, 2, 3, 4, 5, 6];
        int[] bTokens = [13, 17];
        using var expectedA = fixture.Model.Forward(aTokens, [0, 1, 2, 3, 4, 5], -1, serialA);
        using var expectedB = fixture.Model.Forward(bTokens, [0, 1], -1, serialB);
        using var actual = fixture.Model.ForwardBatch([.. aTokens, .. bTokens], [0, 1, 2, 3, 4, 5, 0, 1],
            [new(0, 6, a), new(6, 2, b)], -1);
        AssertRowsEqual(expectedA, 5, actual, 0);
        AssertRowsEqual(expectedB, 1, actual, 1);
        Assert.Equal(6, a.CurrentLength);
        Assert.Equal(2, b.CurrentLength);

        // Mix one decode token with a different sequence's multi-token append; reverse row order.
        using var nextA = fixture.Model.Forward([7], [6], -1, serialA);
        using var nextB = fixture.Model.Forward([18, 19, 20], [2, 3, 4], -1, serialB);
        using var nextBatch = fixture.Model.ForwardBatch([18, 19, 20, 7], [2, 3, 4, 6],
            [new(0, 3, b), new(3, 1, a)], -1);
        AssertRowsEqual(nextA, 0, nextBatch, 1);
        AssertRowsEqual(nextB, 2, nextBatch, 0);
        Assert.Equal(0, a.StagingBytes);
        Assert.Equal(0, b.StagingBytes);
    }

    [Fact]
    public void InvalidBatchDoesNotMutateAnyCache()
    {
        using var fixture = new TinyTransformer();
        using var pool = new PagedKvCacheFactory(2, 2, 8, blockSize: 2, maxTotalTokens: 128);
        using var a = pool.Create(64);
        using var b = pool.Create(64);
        Assert.Throws<ArgumentException>(() => fixture.Model.ForwardBatch([1, 2], [0, 5],
            [new(0, 1, a), new(1, 1, b)], -1));
        Assert.Throws<ArgumentException>(() => fixture.Model.ForwardBatch([1, 2], [0, 0],
            [new(0, 1, a), new(1, 1, a)], -1));
        Assert.Equal(0, a.CurrentLength);
        Assert.Equal(0, b.CurrentLength);
        Assert.Equal(pool.Pool.TotalBlocks, pool.Pool.FreeBlocks);
    }

    private static unsafe void AssertRowsEqual(ITensor expected, int expectedRow, ITensor actual, int actualRow)
    {
        int vocab = expected.Shape[1];
        var e = new ReadOnlySpan<float>((float*)expected.DataPointer + expectedRow * vocab, vocab);
        var a = new ReadOnlySpan<float>((float*)actual.DataPointer + actualRow * vocab, vocab);
        for (int i = 0; i < vocab; i++)
            Assert.InRange(Math.Abs(e[i] - a[i]), 0, 0.00001f);
    }
}

internal sealed class TinyTransformer : IDisposable
{
    private readonly string _path;
    private readonly GgufFile _gguf;
    public TransformerModel Model { get; }
    public TinyTransformer(int kvHeads = 2, int window = 0, int threads = 1)
    {
        var config = new ModelConfig
        {
            Architecture = DotLLM.Core.Configuration.Architecture.Llama, VocabSize = 32, HiddenSize = 32, IntermediateSize = 64,
            NumLayers = 2, NumAttentionHeads = 4, NumKvHeads = kvHeads, HeadDim = 8,
            MaxSequenceLength = 64, SlidingWindowSize = window == 0 ? null : window
        };
        var gguf = new GgufTestData();
        var rng = new Random(42);
        void Add(string name, bool norm, params int[] dims)
        {
            int count = dims.Aggregate(1, (a, b) => a * b);
            byte[] data = new byte[count * sizeof(float)];
            var values = MemoryMarshal.Cast<byte, float>(data.AsSpan());
            for (int i = 0; i < count; i++) values[i] = norm ? 1 : (float)(rng.NextDouble() - 0.5) * 0.2f;
            gguf.AddTensor(name, dims, 0, data);
        }
        Add("token_embd.weight", false, 32, 32);
        Add("output_norm.weight", true, 32);
        Add("output.weight", false, 32, 32);
        for (int layer = 0; layer < 2; layer++)
        {
            string prefix = $"blk.{layer}";
            Add(prefix + ".attn_norm.weight", true, 32);
            Add(prefix + ".attn_q.weight", false, 32, 32);
            Add(prefix + ".attn_k.weight", false, 32, kvHeads * 8);
            Add(prefix + ".attn_v.weight", false, 32, kvHeads * 8);
            Add(prefix + ".attn_output.weight", false, 32, 32);
            Add(prefix + ".ffn_norm.weight", true, 32);
            Add(prefix + ".ffn_gate.weight", false, 32, 64);
            Add(prefix + ".ffn_up.weight", false, 32, 64);
            Add(prefix + ".ffn_down.weight", false, 64, 32);
        }
        _path = gguf.WriteToTempFile();
        _gguf = GgufFile.Open(_path);
        Model = TransformerModel.LoadFromGguf(_gguf, config, new ThreadingConfig(threads));
    }
    public void Dispose() { Model.Dispose(); _gguf.Dispose(); File.Delete(_path); }
}
