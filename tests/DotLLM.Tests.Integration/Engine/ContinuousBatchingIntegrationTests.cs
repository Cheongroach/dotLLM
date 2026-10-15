using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

[Collection("SmallModel")]
public sealed class ContinuousBatchingIntegrationTests(SmallModelFixture fixture)
{
    [Fact]
    public async Task RealModel_ChunkedConcurrentRequestsMatchSerialGreedy()
    {
        using var gguf = GgufFile.Open(fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(2, 2));
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var pages = new PagedKvCacheFactory(config.NumLayers, config.NumKvHeads, config.HeadDim, 4, 256);
        var options = new InferenceOptions { Temperature = 0, MaxTokens = 12 };
        string[] prompts = ["The capital of France is", "Once upon a time in a small village", "Explain why the sky is blue in one short sentence:"];
        var generator = new TextGenerator(model, tokenizer, (cfg, size) => pages.Create(size));
        var expected = prompts.Select(p => generator.Generate(p, options)).ToArray();
        await using var scheduler = new ContinuousBatchScheduler(model, tokenizer, pages, new()
            { MaxSequences = 3, MaxBatchedTokens = 12, PrefillChunkSize = 5 });
        var pending = prompts.Select(p => scheduler.EnqueueAsync(new() { TokenIds = tokenizer.Encode(p), Options = options })).ToArray();
        var loop = Task.Run(() => scheduler.RunLoopAsync(CancellationToken.None));
        var actual = await Task.WhenAll(pending).WaitAsync(TimeSpan.FromMinutes(2));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].GeneratedTokenIds, actual[i].GeneratedTokenIds);
            Assert.Equal(expected[i].Text, actual[i].Text);
        }
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await scheduler.DisposeAsync();
        await loop;
    }
}
