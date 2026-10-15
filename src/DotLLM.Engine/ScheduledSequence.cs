using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

// Mutated only by the scheduler owner. Caller cancellation merely wakes that owner.
internal sealed class ScheduledSequence
{
    public required InferenceRequest Request;
    public required CancellationToken Cancellation;
    public required int Capacity;
    public required int ReservedBlocks;
    public CancellationTokenRegistration Registration;
    public IKvCache? Cache;
    public Channel<GenerationToken>? Stream;
    public readonly TaskCompletionSource<InferenceResponse> Response = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public readonly TaskCompletionSource Released = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public readonly List<int> Generated = [];
    public int Prefilled;
    public int DecodeSteps;
    public long PrefillTicks;
    public long DecodeTicks;
    public long SamplingTicks;
    private SamplerPipeline _sampler = null!;
    private IDecodingConstraint? _constraint;
    private IncrementalDetokenizer _detokenizer = null!;
    private readonly StringBuilder _pendingText = new();
    private readonly StringBuilder _output = new();
    private List<TokenLogprobInfo>? _logprobs;
    public FinishReason? Finished;

    public void Initialize(ITokenizer tokenizer)
    {
        _sampler = new SamplerPipeline(Request.Options);
        _constraint = Request.Constraint?.Clone() ?? Request.Options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(tokenizer, js.Schema),
            ResponseFormat.Regex rx => new RegexConstraint(tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(tokenizer, gr.GbnfGrammar),
            _ => null
        };
        _detokenizer = new IncrementalDetokenizer(tokenizer);
        _logprobs = Request.Options.Logprobs ? [] : null;
    }

    public void Sample(Span<float> logits, ITokenizer tokenizer)
    {
        long start = Stopwatch.GetTimestamp();
        if (_constraint is not null)
            TokenMaskApplier.Apply(logits, _constraint.GetAllowedTokens());
        bool hasCandidate = false;
        foreach (float value in logits)
            if (float.IsFinite(value)) { hasCandidate = true; break; }
        if (!hasCandidate)
            throw new InvalidOperationException("No finite sampling candidates remain.");

        float[]? probabilities = Request.Options.Logprobs ? LogprobsCapture.ComputeLogSoftmax(logits) : null;
        int token;
        TokenLogprobInfo? info = null;
        try
        {
            token = _sampler.Sample(logits, Generated);
            if (probabilities is not null)
                info = LogprobsCapture.BuildInfo(probabilities.AsSpan(0, logits.Length), logits.Length,
                    token, Math.Clamp(Request.Options.TopLogprobs, 0, 20), tokenizer);
        }
        finally
        {
            if (probabilities is not null) ArrayPool<float>.Shared.Return(probabilities);
        }
        _constraint?.Advance(token);
        bool exclude = Request.Options.StopConditions is null && token == tokenizer.EosTokenId;
        if (exclude)
            Finished = FinishReason.Stop;
        else
        {
            Generated.Add(token);
            _detokenizer.Append(token);
            if (Request.Options.StopConditions is { } conditions)
            {
                // Custom conditions retain the documented full-text contract.
                string fullText = _detokenizer.ToString();
                foreach (var condition in conditions)
                {
                    var stop = condition.ShouldStop(token, Generated, fullText);
                    if (stop == StopResult.Continue) continue;
                    exclude = stop == StopResult.Stop;
                    Finished = exclude ? FinishReason.Stop : FinishReason.Length;
                    break;
                }
            }
            if (exclude)
                Generated.RemoveAt(Generated.Count - 1);
            else
            {
                _pendingText.Append(_detokenizer.TakeDelta(flushIncomplete: false));
                if (info.HasValue) _logprobs?.Add(info.Value);
                if (Request.Options.StopConditions is null)
                {
                    // A stop string may end inside a decoded token, or span multiple tokens.
                    int earliest = _pendingText.Length;
                    foreach (string stop in Request.Options.StopSequences)
                        for (int offset = 0; offset < earliest && offset + stop.Length <= _pendingText.Length; offset++)
                        {
                            bool match = true;
                            for (int j = 0; j < stop.Length; j++)
                                if (_pendingText[offset + j] != stop[j]) { match = false; break; }
                            if (match) { earliest = offset; break; }
                        }
                    if (earliest < _pendingText.Length)
                    {
                        _pendingText.Length = earliest;
                        Finished = FinishReason.Stop;
                    }
                }
            }
        }
        if (Finished is null && (Generated.Count >= Request.Options.MaxTokens || Cache!.CurrentLength >= Capacity))
            Finished = FinishReason.Length;

        if (Finished.HasValue && !exclude && Finished != FinishReason.Stop)
            _pendingText.Append(_detokenizer.TakeDelta());

        // Retain a possible stop-string prefix until a later token decides whether it matches.
        int hold = 0;
        if (Finished is null && Request.Options.StopConditions is null)
        {
            foreach (string stop in Request.Options.StopSequences)
            {
                int max = Math.Min(stop.Length - 1, _pendingText.Length);
                for (int length = max; length > hold; length--)
                {
                    bool match = true;
                    for (int i = 0; i < length; i++)
                        if (_pendingText[_pendingText.Length - length + i] != stop[i]) { match = false; break; }
                    if (match) { hold = length; break; }
                }
            }
        }
        string text = _pendingText.ToString(0, _pendingText.Length - hold);
        _pendingText.Remove(0, text.Length);
        _output.Append(text);
        SamplingTicks += Stopwatch.GetTimestamp() - start;
        if (Stream is not null && !Stream.Writer.TryWrite(new GenerationToken(token, text, Finished,
            Finished.HasValue ? Timings() : null, exclude ? null : info,
            Finished.HasValue ? Generated.Count : null)))
            throw new InvalidOperationException("Streaming consumer is too slow; its token buffer is full.");
    }

    public InferenceTimings Timings() => new()
    {
        PrefillTimeMs = PrefillTicks * 1000.0 / Stopwatch.Frequency,
        DecodeTimeMs = DecodeTicks * 1000.0 / Stopwatch.Frequency,
        SamplingTimeMs = SamplingTicks * 1000.0 / Stopwatch.Frequency,
        PrefillTokenCount = Request.TokenIds.Length,
        DecodeTokenCount = DecodeSteps,
        KvCacheBytes = Cache is PagedKvCache paged ? paged.AllocatedBytes : 0
    };

    public void Complete(Exception? error = null)
    {
        var result = error is null ? new InferenceResponse
        {
            Text = _output.ToString(), GeneratedTokenIds = Generated.ToArray(),
            GeneratedTokenCount = Generated.Count, PromptTokenCount = Request.TokenIds.Length,
            FinishReason = Finished ?? FinishReason.Length, Timings = Timings(), Logprobs = _logprobs?.ToArray()
        } : null;
        try { Cache?.Dispose(); }
        catch (Exception cleanupError) { error ??= cleanupError; }
        Cache = null;
        Registration.Dispose();
        if (Stream is not null)
        {
            // Zero-token requests have no Sample call to emit the terminal event.
            if (error is null && Generated.Count == 0 && Finished is null)
                Stream.Writer.TryWrite(new GenerationToken(-1, "", FinishReason.Length, result!.Timings, GeneratedTokenCount: 0));
            Stream.Writer.TryComplete(error);
        }
        else if (error is OperationCanceledException)
            Response.TrySetCanceled(Cancellation.IsCancellationRequested ? Cancellation : new CancellationToken(true));
        else if (error is not null)
            Response.TrySetException(error);
        else
            Response.TrySetResult(result!);
        Released.TrySetResult();
    }
}
