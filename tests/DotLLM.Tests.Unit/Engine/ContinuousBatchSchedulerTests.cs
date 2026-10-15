using System.Collections.Concurrent;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

public sealed class ContinuousBatchSchedulerTests
{
    internal static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);
    internal static InferenceRequest Request(int first = 1, int length = 1, int output = 4) => new()
    {
        TokenIds = Enumerable.Repeat(first, length).ToArray(),
        Options = new InferenceOptions { Temperature = 0, MaxTokens = output }
    };
    private static ContinuousBatchingOptions Options(int sequences = 3, int budget = 8) => new()
        { MaxSequences = sequences, MaxBatchedTokens = budget, PrefillChunkSize = 3 };

    [Fact]
    public async Task PackedPrefillAndDecode_IsolateRequestsAndRespectBudget()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        var a = scheduler.EnqueueAsync(Request(1, 11));
        var b = scheduler.EnqueueAsync(Request(8, 2));
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        var result = await Task.WhenAll(a, b).WaitAsync(Timeout);
        Assert.Equal(new[] { 2, 3, 4, 5 }, result[0].GeneratedTokenIds);
        Assert.Equal(new[] { 9, 10, 11, 12 }, result[1].GeneratedTokenIds);
        Assert.Contains(model.Batches, batch => batch.Length == 2);
        Assert.All(model.Batches, batch => Assert.InRange(batch.Sum(), 1, 8));
        Assert.Contains(model.Batches, batch => batch.Length == 2 && batch.Contains(1) && batch.Any(n => n > 1));
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task LateArrivalJoinsBeforeExistingRequestFinishes()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        Task<InferenceResponse>? late = null;
        var first = scheduler.EnqueueAsync(Request(output: 12));
        model.OnCall = call => { if (call == 2) late = scheduler.EnqueueAsync(Request(15, output: 2)); };
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        await first.WaitAsync(Timeout);
        Assert.NotNull(late);
        Assert.Equal(new[] { 16, 17 }, (await late.WaitAsync(Timeout)).GeneratedTokenIds);
        Assert.Contains(model.Batches, batch => batch.Length == 2);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task ReservationSerializesOversubscribedRequestsWithoutPoolExhaustion()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 8);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        var a = scheduler.EnqueueAsync(Request(1, 3, 5));
        var b = scheduler.EnqueueAsync(Request(9, 3, 5));
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        await Task.WhenAll(a, b).WaitAsync(Timeout);
        Assert.All(model.Batches, batch => Assert.Single(batch));
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task CancelQueuedAndActive_ReclaimsPagesAndAllowsOtherRequest()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options(1));
        using var activeCancel = new CancellationTokenSource();
        using var queuedCancel = new CancellationTokenSource();
        var a = scheduler.EnqueueAsync(Request(output: 20), activeCancel.Token);
        var b = scheduler.EnqueueAsync(Request(10), queuedCancel.Token);
        var c = scheduler.EnqueueAsync(Request(15, output: 2));
        model.OnCall = call => { if (call == 2) { activeCancel.Cancel(); queuedCancel.Cancel(); } };
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => a.WaitAsync(Timeout));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => b.WaitAsync(Timeout));
        Assert.Equal(new[] { 16, 17 }, (await c.WaitAsync(Timeout)).GeneratedTokenIds);
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task ForwardFailureFailsAllRequestsAndReclaimsPages()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options(1));
        var a = scheduler.EnqueueAsync(Request());
        var b = scheduler.EnqueueAsync(Request(10));
        model.OnCall = call => { if (call == 2) throw new InvalidOperationException("injected forward fault"); };
        await Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None)).WaitAsync(Timeout);
        await Assert.ThrowsAsync<InvalidOperationException>(() => a);
        await Assert.ThrowsAsync<InvalidOperationException>(() => b);
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        Assert.Equal(0, scheduler.GetMetrics().QueueDepth);
    }

    [Fact]
    public async Task StopsAtEosAndContextAndHandlesZeroTokens()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 256);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        var eos = scheduler.EnqueueAsync(Request(30));
        var context = scheduler.EnqueueAsync(Request(1, model.Config.MaxSequenceLength, 10));
        var empty = scheduler.EnqueueAsync(Request(output: 0));
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        var results = await Task.WhenAll(eos, context, empty).WaitAsync(Timeout);
        Assert.Empty(results[0].GeneratedTokenIds);
        Assert.Equal(FinishReason.Stop, results[0].FinishReason);
        Assert.Single(results[1].GeneratedTokenIds);
        Assert.Empty(results[2].GeneratedTokenIds);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task StopStringSpanningTokensIsWithheldInStreamAndResponse()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        var request = Request(0, output: 10) with { Options = new() { Temperature = 0, MaxTokens = 10, StopSequences = ["cd"], Logprobs = true } };
        var loop = scheduler.RunLoopAsync(CancellationToken.None);
        var response = await scheduler.EnqueueAsync(request).WaitAsync(Timeout);
        var stream = new List<GenerationToken>();
        await foreach (var token in scheduler.GenerateStreamingAsync(request)) stream.Add(token);
        Assert.Equal("b", response.Text);
        Assert.Equal(response.Text, string.Concat(stream.Select(t => t.Text)));
        Assert.Equal(FinishReason.Stop, stream[^1].FinishReason);
        Assert.Equal(response.GeneratedTokenCount, stream[^1].GeneratedTokenCount);
        Assert.Equal(response.GeneratedTokenCount, response.Logprobs!.Length);
        Assert.DoesNotContain(stream, t => t.Text.Contains('c') || t.Text.Contains('d'));
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task StopInsideDecodedTokenDropsTheRemainingTokenText()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        var tokenizer = new AlphabetTokenizer { Replacements = new() { [2] = "cd!" } };
        await using var scheduler = new ContinuousBatchScheduler(model, tokenizer, pages, Options());
        var loop = scheduler.RunLoopAsync(CancellationToken.None);
        var request = Request(0) with { Options = new() { Temperature = 0, MaxTokens = 4, StopSequences = ["cd"] } };
        var result = await scheduler.EnqueueAsync(request).WaitAsync(Timeout);
        var chunks = new List<string>();
        await foreach (var token in scheduler.GenerateStreamingAsync(request)) chunks.Add(token.Text);
        Assert.Equal("b", result.Text);
        Assert.Equal("b", string.Concat(chunks));
        Assert.Equal(new[] { 1, 2 }, result.GeneratedTokenIds);
        await scheduler.DisposeAsync();
        await loop;
    }

    [Fact]
    public async Task SlowConsumerFailsOnlyItsOwnRequest()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages,
            Options() with { StreamBufferCapacity = 1 });
        await using var reader = scheduler.GenerateStreamingAsync(Request(output: 12)).GetAsyncEnumerator();
        var firstRead = reader.MoveNextAsync().AsTask();
        var other = scheduler.EnqueueAsync(Request(15, output: 5));
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        Assert.True(await firstRead.WaitAsync(Timeout));
        await other.WaitAsync(Timeout);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            while (await reader.MoveNextAsync()) { }
        });
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task DisposalBeforeStartCompletesQueuedRequests()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        var result = scheduler.EnqueueAsync(Request());
        await scheduler.DisposeAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => result.WaitAsync(Timeout));
        await scheduler.DisposeAsync();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => scheduler.EnqueueAsync(Request()));
    }

    [Fact]
    public async Task AdmissionUsesPriorityWithoutPreemptingActiveRequest()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options(1));
        var low = scheduler.EnqueueAsync(Request(1, output: 3));
        var waiting = scheduler.EnqueueAsync(Request(15, output: 1));
        Task<InferenceResponse>? high = null;
        model.OnCall = call => { if (call == 1) high = scheduler.EnqueueAsync(Request(10, output: 1) with { Priority = RequestPriority.High }); };
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        await Task.WhenAll(low, waiting).WaitAsync(Timeout);
        await high!.WaitAsync(Timeout);
        Assert.Equal(new[] { 1, 2, 3, 10, 15 }, model.FirstTokens.ToArray());
        await scheduler.DisposeAsync();
        await loop;
    }

    [Fact]
    public async Task SeedAndConstraintStateAreOwnedByEachRequest()
    {
        async Task<int[][]> Run(int count, IDecodingConstraint? constraint = null)
        {
            using var model = new CountingModel { UniformLogits = true };
            using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
            await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
            var request = Request() with { Options = new() { Temperature = 1, MaxTokens = 8, Seed = 123 }, Constraint = constraint };
            var pending = Enumerable.Range(0, count).Select(_ => scheduler.EnqueueAsync(request)).ToArray();
            var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
            var responses = await Task.WhenAll(pending).WaitAsync(Timeout);
            await scheduler.DisposeAsync();
            await loop;
            return responses.Select(r => r.GeneratedTokenIds).ToArray();
        }
        int[] serial = (await Run(1))[0];
        Assert.All(await Run(3), tokens => Assert.Equal(serial, tokens));
        var constraint = new SteppingConstraint();
        Assert.All(await Run(3, constraint), tokens => Assert.Equal(Enumerable.Range(2, 8), tokens));
        Assert.Equal(0, constraint.Step);
    }

    [Fact]
    public async Task DisposingStreamWaitsForInFlightForwardAndReturnsItsPages()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages, Options());
        using var release = new ManualResetEventSlim();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        model.OnCall = call => { if (call == 2) { entered.SetResult(); Assert.True(release.Wait(Timeout)); } };
        var reader = scheduler.GenerateStreamingAsync(Request(output: 20)).GetAsyncEnumerator();
        var first = reader.MoveNextAsync().AsTask();
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        try
        {
            Assert.True(await first.WaitAsync(Timeout));
            await entered.Task.WaitAsync(Timeout);
            var disposing = reader.DisposeAsync().AsTask();
            Assert.False(disposing.IsCompleted);
            var other = scheduler.EnqueueAsync(Request(15, output: 2));
            release.Set();
            await disposing.WaitAsync(Timeout);
            await other.WaitAsync(Timeout);
            Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        }
        finally { release.Set(); await reader.DisposeAsync(); }
        await scheduler.DisposeAsync();
        await loop;
    }

    [Fact]
    public async Task CacheDisposalFailureCompletesRequestWithoutHangingShutdown()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 64);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(),
            size => new ThrowOnDisposeCache(pages.Create(size)), 2, pages.Pool.TotalBlocks, Options());
        var pending = scheduler.EnqueueAsync(Request(output: 1));
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        await Assert.ThrowsAsync<InvalidOperationException>(() => pending.WaitAsync(Timeout));
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop.WaitAsync(Timeout);
    }

    [Fact]
    public async Task InvalidRequestsAndFullQueueAreRejectedBeforeCompute()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 16);
        await using var scheduler = new ContinuousBatchScheduler(model, new AlphabetTokenizer(), pages,
            Options() with { MaxQueuedRequests = 1 });
        await Assert.ThrowsAsync<ArgumentException>(() => scheduler.EnqueueAsync(Request(length: 0)));
        await Assert.ThrowsAsync<ArgumentException>(() => scheduler.EnqueueAsync(Request(first: 32)));
        await Assert.ThrowsAsync<ArgumentException>(() => scheduler.EnqueueAsync(Request(output: 30)));
        await Assert.ThrowsAsync<NotSupportedException>(() => scheduler.EnqueueAsync(Request() with { AdapterId = "adapter" }));
        var waiting = scheduler.EnqueueAsync(Request());
        await Assert.ThrowsAsync<InvalidOperationException>(() => scheduler.EnqueueAsync(Request()));
        await scheduler.DisposeAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        Assert.Empty(model.Batches);
    }
}

internal sealed class AlphabetTokenizer : ITokenizer
{
    public Dictionary<int, string>? Replacements;
    public int VocabSize => 32;
    public int BosTokenId => 0;
    public int EosTokenId => 31;
    public int[] Encode(string text) => text.Select(c => (c - 'a') % 31).ToArray();
    public string Decode(ReadOnlySpan<int> ids) => string.Concat(ids.ToArray().Where(t => t != EosTokenId).Select(DecodeToken));
    public string DecodeToken(int token) => Replacements?.GetValueOrDefault(token) ?? (token == EosTokenId ? "" : ((char)('a' + token)).ToString());
    public int CountTokens(string text) => text.Length;
}

internal sealed class CountingModel : IBatchedModel
{
    public ModelConfig Config { get; } = new()
    {
        Architecture = Architecture.Llama, VocabSize = 32, HiddenSize = 2, IntermediateSize = 4,
        NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 2, MaxSequenceLength = 64
    };
    public long ComputeMemoryBytes => 0;
    public readonly ConcurrentQueue<int[]> Batches = new();
    public readonly ConcurrentQueue<int> FirstTokens = new();
    public bool UniformLogits;
    public Action<int>? OnCall;
    private int _calls;
    public unsafe ITensor ForwardBatch(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions,
        ReadOnlySpan<SequenceBatch> sequences, int deviceId)
    {
        SequenceBatch.Validate(tokens, positions, sequences, Config);
        FirstTokens.Enqueue(tokens[0]);
        Batches.Enqueue(sequences.ToArray().Select(s => s.TokenCount).ToArray());
        using var data = UnmanagedTensor.Allocate(new TensorShape(tokens.Length, 2), DType.Float32);
        var values = new Span<float>((void*)data.DataPointer, tokens.Length * 2);
        for (int i = 0; i < tokens.Length; i++) { values[2 * i] = tokens[i]; values[2 * i + 1] = positions[i]; }
        foreach (var s in sequences)
        {
            var projection = new TensorRef(s.TokenCount, 2, DType.Float32, -1, data.DataPointer + s.TokenOffset * 8);
            s.Cache.Update(projection, projection, positions.Slice(s.TokenOffset, s.TokenCount), 0);
        }
        OnCall?.Invoke(++_calls);
        var logits = UnmanagedTensor.Allocate(new TensorShape(sequences.Length, 32), DType.Float32);
        var output = new Span<float>((void*)logits.DataPointer, sequences.Length * 32);
        output.Fill(UniformLogits ? 0 : -100);
        for (int i = 0; i < sequences.Length; i++)
        {
            int last = tokens[sequences[i].TokenOffset + sequences[i].TokenCount - 1];
            if (!UniformLogits) output[i * 32 + (last + 1) % 32] = 10;
        }
        return logits;
    }
    public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => throw new NotSupportedException("Must use packed forward.");
    public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? cache) => throw new NotSupportedException("Must use packed forward.");
    public void Dispose() { }
}

internal sealed class SteppingConstraint : IDecodingConstraint
{
    public int Step;
    public void Advance(int tokenId) => Step++;
    public TokenMask GetAllowedTokens() { var mask = new TokenMask(32); mask.Allow(2 + Step); return mask; }
    public bool IsComplete() => false;
    public IDecodingConstraint Clone() => new SteppingConstraint { Step = Step };
    public void Reset() => Step = 0;
}

internal sealed class ThrowOnDisposeCache(IKvCache inner) : IKvCache
{
    public int CurrentLength => inner.CurrentLength;
    public int MaxLength => inner.MaxLength;
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
    public ITensor GetKeys(int layerIndex) => inner.GetKeys(layerIndex);
    public ITensor GetValues(int layerIndex) => inner.GetValues(layerIndex);
    public TensorRef GetKeysRef(int layerIndex) => inner.GetKeysRef(layerIndex);
    public TensorRef GetValuesRef(int layerIndex) => inner.GetValuesRef(layerIndex);
    public void Rollback(int length) => inner.Rollback(length);
    public void Dispose() { inner.Dispose(); throw new InvalidOperationException("Injected cache disposal fault"); }
}
