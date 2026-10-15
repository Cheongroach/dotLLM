using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Single-owner scheduler with packed forwards, decode-first token budgets and chunked prefill.
/// Owns request caches; the caller owns the model and page pool and must dispose them after this scheduler.
/// Priority affects queued admission only. No eviction or active-sequence preemption is performed.
/// </summary>
public sealed class ContinuousBatchScheduler : IScheduler, IDisposable, IAsyncDisposable
{
    private readonly IBatchedModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly Func<int, IKvCache> _cacheFactory;
    private readonly int _blockSize;
    private readonly int _totalBlocks;
    private readonly ContinuousBatchingOptions _options;
    private readonly object _gate = new();
    private readonly List<ScheduledSequence> _incoming = [];
    private readonly List<ScheduledSequence> _waiting = [];
    private readonly List<ScheduledSequence> _active = [];
    private readonly Channel<byte> _wake = Channel.CreateBounded<byte>(new BoundedChannelOptions(1)
    { SingleReader = true, FullMode = BoundedChannelFullMode.DropWrite });
    private readonly CancellationTokenSource _stop = new();
    private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private bool _accepting = true;
    private bool _started;
    private int _queued;
    private int _activeCount;
    private int _reserved;
    private int _prefillCursor;
    private Task? _disposeTask;

    /// <summary>Creates a scheduler using an exclusively assigned CPU page pool.</summary>
    public ContinuousBatchScheduler(IBatchedModel model, ITokenizer tokenizer,
        PagedKvCacheFactory cacheFactory, ContinuousBatchingOptions? options = null)
        : this(model, tokenizer, cacheFactory.Create, cacheFactory.Pool.BlockSize,
            cacheFactory.Pool.TotalBlocks, options) { }

    /// <summary>
    /// Creates a scheduler with a backend-specific cache factory and exclusive pool capacity.
    /// Custom sampler steps, processors and stop conditions must be request-owned or immutable.
    /// </summary>
    public ContinuousBatchScheduler(IBatchedModel model, ITokenizer tokenizer,
        Func<int, IKvCache> cacheFactory, int blockSize, int totalBlocks,
        ContinuousBatchingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(cacheFactory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalBlocks);
        _model = model;
        _tokenizer = tokenizer;
        _cacheFactory = cacheFactory;
        _blockSize = blockSize;
        _totalBlocks = totalBlocks;
        _options = options ?? new();
        _options.Validate();
    }

    /// <inheritdoc/>
    public Task<InferenceResponse> EnqueueAsync(InferenceRequest request, CancellationToken cancellationToken = default)
        => Submit(request, cancellationToken, streaming: false).Response.Task;

    /// <summary>Streams one request. Disposing enumeration cancels it and waits for KV reclamation.</summary>
    public async IAsyncEnumerable<GenerationToken> GenerateStreamingAsync(InferenceRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var sequence = Submit(request, cancellation.Token, streaming: true);
        try
        {
            await foreach (var token in sequence.Stream!.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
                yield return token;
        }
        finally
        {
            cancellation.Cancel();
            await sequence.Released.Task.ConfigureAwait(false);
        }
    }

    private ScheduledSequence Submit(InferenceRequest request, CancellationToken ct, bool streaming)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.TokenIds);
        ArgumentNullException.ThrowIfNull(request.Options);
        ct.ThrowIfCancellationRequested();
        if (request.TokenIds.Length == 0 || request.TokenIds.Length > _model.Config.MaxSequenceLength)
            throw new ArgumentException("Prompt must contain between 1 and the model context limit tokens.");
        foreach (int token in request.TokenIds)
            if ((uint)token >= (uint)_model.Config.VocabSize) throw new ArgumentException("Invalid prompt token.");
        ArgumentOutOfRangeException.ThrowIfNegative(request.Options.MaxTokens);
        if (request.AdapterId is not null) throw new NotSupportedException("Per-request adapters are not supported by batched forwards.");
        if (request.Options.StopSequences.Any(string.IsNullOrEmpty)) throw new ArgumentException("Stop strings must be nonempty.");
        int capacity = (int)Math.Min(_model.Config.MaxSequenceLength,
            (long)request.TokenIds.Length + Math.Max(0, request.Options.MaxTokens - 1));
        int blocks = request.Options.MaxTokens == 0 ? 0 : (int)(((long)capacity + _blockSize - 1) / _blockSize);
        if (blocks > _totalBlocks) throw new ArgumentException("Request exceeds the page pool capacity.");
        var sequence = new ScheduledSequence
        {
            Request = request with { TokenIds = (int[])request.TokenIds.Clone(), Options = request.Options with
                { StopSequences = request.Options.StopSequences.ToArray() } },
            Cancellation = ct, Capacity = capacity, ReservedBlocks = blocks,
            Stream = streaming ? Channel.CreateBounded<GenerationToken>(new BoundedChannelOptions(_options.StreamBufferCapacity)
                { SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait }) : null
        };
        lock (_gate)
        {
            if (!_accepting) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));
            if (_queued >= _options.MaxQueuedRequests) throw new InvalidOperationException("Scheduler queue is full.");
            sequence.Registration = ct.Register(Wake);
            _incoming.Add(sequence);
            Interlocked.Increment(ref _queued);
        }
        Wake();
        return sequence;
    }

    private void Wake() => _wake.Writer.TryWrite(0);

    /// <inheritdoc/>
    public async Task RunLoopAsync(CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (_started) throw new InvalidOperationException("The scheduler loop can only run once.");
            if (!_accepting) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));
            _started = true;
        }
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _stop.Token);
        Exception failure = new OperationCanceledException("Scheduler stopped.", linked.Token);
        // Buffers live for the loop; no per-iteration token/position/segment allocations.
        var tokens = new int[_options.MaxBatchedTokens];
        var positions = new int[_options.MaxBatchedTokens];
        var segments = new SequenceBatch[_options.MaxSequences];
        var scheduled = new ScheduledSequence[_options.MaxSequences];
        try
        {
            while (!linked.IsCancellationRequested)
            {
                DrainIncoming();
                RemoveCanceled();
                Admit();
                Volatile.Write(ref _activeCount, _active.Count);
                if (_active.Count == 0)
                {
                    await _wake.Reader.ReadAsync(linked.Token).ConfigureAwait(false);
                    continue;
                }
                RunIteration(tokens, positions, segments, scheduled);
            }
        }
        catch (Exception error) { failure = error; }
        finally
        {
            lock (_gate) _accepting = false;
            DrainIncoming();
            foreach (var sequence in _active) sequence.Complete(failure);
            foreach (var sequence in _waiting) sequence.Complete(failure);
            _active.Clear();
            _waiting.Clear();
            Volatile.Write(ref _activeCount, 0);
            Volatile.Write(ref _queued, 0);
            _stopped.TrySetResult();
        }
    }

    private void DrainIncoming()
    {
        lock (_gate)
        {
            _waiting.AddRange(_incoming);
            _incoming.Clear();
        }
    }

    private void RemoveCanceled()
    {
        for (int i = _waiting.Count - 1; i >= 0; i--)
        {
            var sequence = _waiting[i];
            if (!sequence.Cancellation.IsCancellationRequested) continue;
            _waiting.RemoveAt(i);
            Interlocked.Decrement(ref _queued);
            sequence.Complete(new OperationCanceledException(sequence.Cancellation));
        }
        for (int i = _active.Count - 1; i >= 0; i--)
            if (_active[i].Cancellation.IsCancellationRequested)
                Finish(_active[i], new OperationCanceledException(_active[i].Cancellation));
    }

    private void Admit()
    {
        while (_waiting.Count > 0 && _active.Count < _options.MaxSequences)
        {
            // Stable priority ordering. Do not let new small requests bypass an older capacity waiter.
            int best = 0;
            for (int i = 1; i < _waiting.Count; i++)
                if (_waiting[i].Request.Priority > _waiting[best].Request.Priority) best = i;
            var sequence = _waiting[best];
            if (_reserved + sequence.ReservedBlocks > _totalBlocks) break;
            _waiting.RemoveAt(best);
            Interlocked.Decrement(ref _queued);
            if (sequence.Request.Options.MaxTokens == 0) { sequence.Complete(); continue; }
            try
            {
                sequence.Initialize(_tokenizer);
                sequence.Cache = _cacheFactory(sequence.Capacity);
                _reserved += sequence.ReservedBlocks;
                _active.Add(sequence);
            }
            catch (Exception error) { sequence.Complete(error); }
        }
    }

    private unsafe void RunIteration(int[] tokens, int[] positions, SequenceBatch[] segments, ScheduledSequence[] scheduled)
    {
        int used = 0, count = 0;
        foreach (var sequence in _active)
        {
            if (sequence.Prefilled != sequence.Request.TokenIds.Length) continue;
            tokens[used] = sequence.Generated[^1];
            positions[used] = sequence.Cache!.CurrentLength;
            scheduled[count] = sequence;
            segments[count++] = new SequenceBatch(used++, 1, sequence.Cache);
        }
        int start = _prefillCursor % _active.Count;
        for (int i = 0; i < _active.Count && used < tokens.Length; i++)
        {
            var sequence = _active[(start + i) % _active.Count];
            int remaining = sequence.Request.TokenIds.Length - sequence.Prefilled;
            if (remaining == 0) continue;
            int take = Math.Min(remaining, Math.Min(_options.PrefillChunkSize, tokens.Length - used));
            sequence.Request.TokenIds.AsSpan(sequence.Prefilled, take).CopyTo(tokens.AsSpan(used));
            for (int j = 0; j < take; j++) positions[used + j] = sequence.Prefilled + j;
            scheduled[count] = sequence;
            segments[count++] = new SequenceBatch(used, take, sequence.Cache!);
            used += take;
        }
        _prefillCursor = (start + 1) % _active.Count;
        long startTime = Stopwatch.GetTimestamp();
        using ITensor logits = _model.ForwardBatch(tokens.AsSpan(0, used), positions.AsSpan(0, used),
            segments.AsSpan(0, count), deviceId: -1);
        long ticks = Stopwatch.GetTimestamp() - startTime;
        if (logits.DType != DType.Float32 || logits.DeviceId != -1 || logits.Shape.Rank != 2 ||
            logits.Shape[0] != count || logits.Shape[1] != _model.Config.VocabSize)
            throw new InvalidOperationException("ForwardBatch must return CPU Float32 [sequences, vocab] logits.");
        for (int i = 0; i < count; i++)
        {
            var sequence = scheduled[i];
            try
            {
                sequence.Cancellation.ThrowIfCancellationRequested();
                if (sequence.Prefilled < sequence.Request.TokenIds.Length)
                {
                    sequence.Prefilled += segments[i].TokenCount;
                    sequence.PrefillTicks += ticks;
                    if (sequence.Prefilled < sequence.Request.TokenIds.Length) continue;
                }
                else { sequence.DecodeTicks += ticks; sequence.DecodeSteps++; }
                sequence.Sample(new Span<float>((float*)logits.DataPointer + (long)i * _model.Config.VocabSize,
                    _model.Config.VocabSize), _tokenizer);
                if (sequence.Finished.HasValue) Finish(sequence);
            }
            catch (Exception error) { Finish(sequence, error); }
        }
        Array.Clear(scheduled, 0, count);
        Array.Clear(segments, 0, count);
        Volatile.Write(ref _activeCount, _active.Count);
    }

    private void Finish(ScheduledSequence sequence, Exception? error = null)
    {
        _active.Remove(sequence);
        _reserved -= sequence.ReservedBlocks;
        sequence.Complete(error);
    }

    /// <inheritdoc/>
    public SchedulerMetrics GetMetrics() => new(Volatile.Read(ref _activeCount), Volatile.Read(ref _queued), 0);

    /// <summary>Stops compute and completes/cancels every request before returning. Does not dispose the model or pool.</summary>
    public ValueTask DisposeAsync()
    {
        lock (_gate) return new ValueTask(_disposeTask ??= StopAsync());
    }

    private async Task StopAsync()
    {
        bool started;
        lock (_gate)
        {
            _accepting = false;
            started = _started;
        }
        _stop.Cancel();
        Wake();
        if (!started)
        {
            DrainIncoming();
            foreach (var sequence in _waiting) sequence.Complete(new OperationCanceledException("Scheduler disposed before start."));
            _waiting.Clear();
            Volatile.Write(ref _queued, 0);
            _stopped.TrySetResult();
        }
        await _stopped.Task.ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
