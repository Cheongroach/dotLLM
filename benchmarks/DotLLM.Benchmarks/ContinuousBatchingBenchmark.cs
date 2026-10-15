using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>Finite arrival-trace benchmark. Emits one JSON result per active-sequence limit.</summary>
public static class ContinuousBatchingBenchmark
{
    /// <summary>Runs --continuous MODEL [cpu|gpu] [concurrency=4] [requests=16] [tokens=32] [arrivalMs=0] [threads=4].</summary>
    public static async Task RunAsync(string[] args)
    {
        if (args.Length < 1) throw new ArgumentException("Supply a GGUF model path.");
        string backend = args.ElementAtOrDefault(1) ?? "cpu";
        if (backend is not ("cpu" or "gpu")) throw new ArgumentException("Backend must be cpu or gpu.");
        int Read(int index, int fallback) => args.Length > index ? int.Parse(args[index]) : fallback;
        int concurrency = Read(2, 4), requests = Read(3, 16), tokens = Read(4, 32), arrivalMs = Read(5, 0), threads = Read(6, 4);
        if (concurrency < 1 || requests < 1 || tokens < 1 || arrivalMs < 0 || threads < 1) throw new ArgumentException("Invalid benchmark dimensions.");
        using var gguf = GgufFile.Open(args[0]);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using IBatchedModel model = backend == "gpu" ? CudaTransformerModel.LoadFromGguf(gguf, config)
            : TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(threads, threads));
        string[] prompts = ["The capital of France is", "Once upon a time in a small village", "Explain why the sky is blue in one short sentence:", "Write a short story about a scientist who discovers a new planet:"];
        var inputs = Enumerable.Range(0, requests).Select(i => new InferenceRequest
            { TokenIds = tokenizer.Encode(prompts[i % prompts.Length]), Options = new() { Temperature = 0, MaxTokens = tokens } }).ToArray();
        int maxCapacity = inputs.Max(r => Math.Min(config.MaxSequenceLength, r.TokenIds.Length + tokens));
        int poolTokens = checked(concurrency * ((maxCapacity + 15) / 16 * 16));
        foreach (int limit in new[] { 1, concurrency }.Distinct())
        {
            using var cpuPages = backend == "cpu" ? new PagedKvCacheFactory(config.NumLayers, config.NumKvHeads, config.HeadDim, 16, poolTokens) : null;
            using var gpuPages = model is CudaTransformerModel cuda ? cuda.CreatePagedKvCacheFactory(poolTokens) : null;
            Func<int, IKvCache> factory = gpuPages is not null ? gpuPages.Create : cpuPages!.Create;
            var measured = new MeasuredModel(model);
            await using var scheduler = new ContinuousBatchScheduler(measured, tokenizer, factory, 16,
                gpuPages?.TotalBlocks ?? cpuPages!.Pool.TotalBlocks, new()
                {
                    MaxSequences = limit, MaxBatchedTokens = Math.Max(128, checked(concurrency + 1)), PrefillChunkSize = 32,
                    MaxQueuedRequests = requests + 2, StreamBufferCapacity = Math.Max(256, tokens + 1)
                });
            var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
            // Warm both decode and multi-row GEMM paths before measuring.
            await Task.WhenAll(Enumerable.Range(0, limit).Select(i => scheduler.EnqueueAsync(inputs[i % inputs.Length])));
            measured.Reset();
            long start = Stopwatch.GetTimestamp();
            var results = await Task.WhenAll(inputs.Select((request, i) => Observe(request, i)));
            double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
            await scheduler.DisposeAsync();
            await loop;
            double[] first = results.Where(r => r.FirstMs.HasValue).Select(r => r.FirstMs!.Value).Order().ToArray();
            double[] intervals = results.SelectMany(r => r.Intervals).Order().ToArray();
            int free = gpuPages?.FreeBlocks ?? cpuPages!.Pool.FreeBlocks;
            int total = gpuPages?.TotalBlocks ?? cpuPages!.Pool.TotalBlocks;
            if (free != total) throw new InvalidOperationException("KV pages leaked during benchmark.");
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                backend, model = Path.GetFileName(args[0]), threads, requests, max_tokens = tokens, arrival_ms = arrivalMs,
                max_sequences = limit, seconds, generated_tokens = results.Sum(r => r.Tokens),
                tokens_per_second = results.Sum(r => r.Tokens) / seconds,
                ttft_p50_ms = Percentile(first, 0.50), ttft_p95_ms = Percentile(first, 0.95),
                itl_p50_ms = Percentile(intervals, 0.50), itl_p95_ms = Percentile(intervals, 0.95),
                forwards = measured.Calls, max_batch_sequences = measured.MaxSequences,
                mean_batch_sequences = (double)measured.SequenceRows / measured.Calls, kv_pages_reclaimed = free == total,
                runtime = Environment.Version.ToString(), logical_processors = Environment.ProcessorCount
            }));

            async Task<Observation> Observe(InferenceRequest request, int index)
            {
                if (arrivalMs > 0) await Task.Delay(checked(index * arrivalMs));
                long submitted = Stopwatch.GetTimestamp(), previous = 0;
                double? ttft = null;
                var intervals = new List<double>();
                int count = 0;
                await foreach (var token in scheduler.GenerateStreamingAsync(request))
                {
                    long now = Stopwatch.GetTimestamp();
                    // Final EOS is excluded from generated token counts and latency samples.
                    bool generated = token.GeneratedTokenCount is null || token.GeneratedTokenCount > count;
                    if (generated)
                    {
                        ttft ??= Stopwatch.GetElapsedTime(submitted, now).TotalMilliseconds;
                        if (previous != 0) intervals.Add(Stopwatch.GetElapsedTime(previous, now).TotalMilliseconds);
                        previous = now;
                        count++;
                    }
                    if (token.GeneratedTokenCount is { } finalCount) count = finalCount;
                }
                return new Observation(count, ttft, intervals);
            }
        }
    }

    private static double? Percentile(double[] sorted, double percentile)
        => sorted.Length == 0 ? null : sorted[(int)Math.Ceiling(percentile * sorted.Length) - 1];
    private sealed record Observation(int Tokens, double? FirstMs, List<double> Intervals);

    private sealed class MeasuredModel(IBatchedModel inner) : IBatchedModel
    {
        public long Calls, SequenceRows;
        public int MaxSequences;
        public ModelConfig Config => inner.Config;
        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;
        public void Reset() { Calls = SequenceRows = 0; MaxSequences = 0; }
        public ITensor ForwardBatch(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, ReadOnlySpan<SequenceBatch> sequences, int deviceId)
        {
            Calls++;
            SequenceRows += sequences.Length;
            MaxSequences = Math.Max(MaxSequences, sequences.Length);
            return inner.ForwardBatch(tokens, positions, sequences, deviceId);
        }
        public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, int deviceId) => inner.Forward(tokens, positions, deviceId);
        public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, int deviceId, IKvCache? cache) => inner.Forward(tokens, positions, deviceId, cache);
        public void Dispose() { }
    }
}
