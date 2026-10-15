using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public sealed class CudaContinuousBatchingTests
{
    [SkippableFact]
    public async Task RealModel_ConcurrentGreedyMatchesSerialAndReclaimsPages()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GPU_TEST_MODEL");
        Skip.If(path is null || !File.Exists(path), "Set DOTLLM_GPU_TEST_MODEL to a local GGUF model");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var pages = model.CreatePagedKvCacheFactory(512, 4);
        var options = new InferenceOptions { Temperature = 0, MaxTokens = 12 };
        string[] prompts = ["The capital of France is", "Once upon a time in a small village", "Explain why the sky is blue in one short sentence:"];
        var generator = new TextGenerator(model, tokenizer, (cfg, size) => pages.Create(size));
        var expected = prompts.Select(p => generator.Generate(p, options)).ToArray();
        await using var scheduler = new ContinuousBatchScheduler(model, tokenizer, pages.Create,
            pages.BlockSize, pages.TotalBlocks, new() { MaxSequences = 3, MaxBatchedTokens = 12, PrefillChunkSize = 5 });
        var pending = prompts.Select(p => scheduler.EnqueueAsync(new() { TokenIds = tokenizer.Encode(p), Options = options })).ToArray();
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        var actual = await Task.WhenAll(pending).WaitAsync(TimeSpan.FromMinutes(2));
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].GeneratedTokenIds, actual[i].GeneratedTokenIds);
        Assert.Equal(pages.TotalBlocks, pages.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop;
    }
}
