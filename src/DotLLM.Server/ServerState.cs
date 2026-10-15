using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Shared server state: loaded model, concurrency control, mutable configuration.
/// Endpoints access live model/generator instances through this object.
/// May start "bare" (no model loaded) when <c>dotllm serve</c> is run without a model argument.
/// </summary>
public sealed class ServerState : IDisposable
{
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private readonly object _drainLock = new();
    private int _inflight;
    private TaskCompletionSource? _drained;
    private bool _disposed;

    /// <summary>Single compute owner for continuous batching, when enabled.</summary>
    public ContinuousBatchScheduler? Scheduler { get; set; }


    /// <summary>Server startup options (updated on model swap).</summary>
    public required ServerOptions Options { get; set; }

    /// <summary>Model configuration (null when no model loaded).</summary>
    public ModelConfig? Config { get; set; }

    /// <summary>Tool call parser for the loaded model.</summary>
    public IToolCallParser? ToolCallParser { get; set; }

    /// <summary>KV-cache configuration.</summary>
    public KvCacheConfig KvCacheConfig { get; set; }

    /// <summary>KV-cache factory for the loaded model/device.</summary>
    public Func<ModelConfig, int, IKvCache>? KvCacheFactory { get; set; }

    /// <summary>Paged KV-cache factory (non-null when paged mode is active). Owns the shared block pool.</summary>
    public PagedKvCacheFactory? PagedFactory { get; set; }

    /// <summary>GPU page pool, disposed after the scheduler and before its model/context.</summary>
    public DotLLM.Cuda.CudaPagedKvCacheFactory? CudaPagedFactory { get; set; }

    /// <summary>Prefix cache for prompt caching (null when disabled).</summary>
    public PrefixCache? PrefixCache { get; set; }

    /// <summary>Whether a model is loaded and ready to accept requests.</summary>
    public bool IsReady { get; set; }

    // ── Live instances (nullable — null when no model loaded) ──

    /// <summary>Currently loaded model.</summary>
    public IModel? Model { get; set; }

    /// <summary>Tokenizer for the loaded model.</summary>
    public ITokenizer? Tokenizer { get; set; }

    /// <summary>Chat template for the loaded model.</summary>
    public IChatTemplate? ChatTemplate { get; set; }

    /// <summary>Text generator wired to the current model.</summary>
    public TextGenerator? Generator { get; set; }

    /// <summary>Mutable sampling parameter defaults (changeable from the UI).</summary>
    public SamplingDefaults SamplingDefaults { get; set; } = new();

    /// <summary>Path of the currently loaded GGUF file.</summary>
    public string LoadedModelPath { get; set; } = "";

    /// <summary>Open GGUF file handle (disposed on model swap).</summary>
    public GgufFile? CurrentGguf { get; set; }

    /// <summary>Draft model for speculative decoding (null when disabled).</summary>
    public IModel? DraftModel { get; set; }

    /// <summary>Path of the loaded draft model GGUF file.</summary>
    public string DraftModelPath { get; set; } = "";

    /// <summary>Open draft GGUF file handle (disposed on model swap).</summary>
    public GgufFile? DraftGguf { get; set; }

    /// <summary>
    /// Holds a model lease for the complete request, including prompt conversion and streaming.
    /// Batched requests overlap; legacy generators retain exclusive access.
    /// </summary>
    public async Task ExecuteAsync(Func<Task> work, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        if (_disposed) { _requestGate.Release(); throw new ObjectDisposedException(nameof(ServerState)); }
        bool concurrent = Scheduler is not null;
        lock (_drainLock)
        {
            if (_inflight++ == 0) _drained = new(TaskCreationOptions.RunContinuationsAsynchronously);
        }
        if (concurrent) _requestGate.Release();
        try { await work(); }
        finally
        {
            lock (_drainLock)
                if (--_inflight == 0) _drained!.TrySetResult();
            if (!concurrent) _requestGate.Release();
        }
    }

    private Task DrainAsync()
    {
        lock (_drainLock) return _inflight == 0 ? Task.CompletedTask : _drained!.Task;
    }

    /// <summary>Generates using the selected execution mode. Call within ExecuteAsync.</summary>
    public Task<InferenceResponse> GenerateAsync(string prompt, InferenceOptions options, CancellationToken ct)
        => Scheduler is { } scheduler
            ? scheduler.EnqueueAsync(ToRequest(prompt, options), ct)
            : Task.FromResult(Generator!.Generate(prompt, options));

    /// <summary>Streams using the selected execution mode. Enumerate within ExecuteAsync.</summary>
    public IAsyncEnumerable<GenerationToken> GenerateStreamingAsync(string prompt, InferenceOptions options, CancellationToken ct)
        => Scheduler is { } scheduler
            ? scheduler.GenerateStreamingAsync(ToRequest(prompt, options), ct)
            : Generator!.GenerateStreamingTokensAsync(prompt, options, ct);

    private InferenceRequest ToRequest(string prompt, InferenceOptions options)
    {
        int[] tokens = Tokenizer!.Encode(prompt);
        return new InferenceRequest { TokenIds = tokens.Length == 0 ? [Tokenizer.BosTokenId] : tokens, Options = options };
    }

    /// <summary>
    /// Loads or swaps a model under the request gate.
    /// Blocks new requests during the swap, disposes the old model (if any), and runs the load action.
    /// </summary>
    public async Task SwapModelAsync(Func<Task> loadAction, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            await DrainAsync();
            IsReady = false;
            if (Scheduler is not null) await Scheduler.DisposeAsync();
            Scheduler = null;
            Generator = null;
            PrefixCache?.Dispose();
            PrefixCache = null;
            PagedFactory?.Dispose();
            PagedFactory = null;
            CudaPagedFactory?.Dispose();
            CudaPagedFactory = null;
            DraftModel?.Dispose();
            DraftModel = null;
            DraftGguf?.Dispose();
            DraftGguf = null;
            Model?.Dispose();
            Model = null;
            CurrentGguf?.Dispose();
            CurrentGguf = null;

            await loadAction();
            IsReady = true;
        }
        finally { _requestGate.Release(); }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _requestGate.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            IsReady = false;
            // Abort queued/active inference on shutdown, then wait for HTTP leases to unwind.
            Scheduler?.Dispose();
            DrainAsync().GetAwaiter().GetResult();
            PrefixCache?.Dispose();
            PagedFactory?.Dispose();
            CudaPagedFactory?.Dispose();
            DraftModel?.Dispose();
            DraftGguf?.Dispose();
            Model?.Dispose();
            CurrentGguf?.Dispose();
        }
        finally { _requestGate.Release(); }
    }
}

/// <summary>
/// Immutable sampling parameter defaults that can be changed from the UI.
/// These serve as defaults when the per-request body does not specify a value.
/// Replaced atomically via <c>with</c> expressions to avoid torn reads.
/// </summary>
public sealed record SamplingDefaults
{
    /// <summary>Sampling temperature. 0 = greedy.</summary>
    public float Temperature { get; init; } = 0.0f;

    /// <summary>Top-P (nucleus) sampling threshold.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Top-K sampling. 0 = disabled.</summary>
    public int TopK { get; init; }

    /// <summary>Min-P sampling threshold. 0 = disabled.</summary>
    public float MinP { get; init; }

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Maximum tokens to generate per response.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed for reproducibility. Null = non-deterministic.</summary>
    public int? Seed { get; init; }
}
