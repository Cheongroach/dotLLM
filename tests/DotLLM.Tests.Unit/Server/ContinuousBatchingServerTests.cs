using System.Net.Http.Json;
using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Server;
using DotLLM.Tests.Unit.Engine;
using DotLLM.Tokenizers.ChatTemplates;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class ContinuousBatchingServerTests
{
    [Fact]
    public async Task ModelSwapWaitsForAllConcurrentRequestLeases()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        using var state = CreateState(model, pages);
        var firstEntered = Signal();
        var secondEntered = Signal();
        var leaveFirst = Signal();
        var leaveSecond = Signal();
        var first = state.ExecuteAsync(async () => { firstEntered.SetResult(); await leaveFirst.Task; }, default);
        var second = state.ExecuteAsync(async () => { secondEntered.SetResult(); await leaveSecond.Task; }, default);
        await Task.WhenAll(firstEntered.Task, secondEntered.Task).WaitAsync(ContinuousBatchSchedulerTests.Timeout);
        bool swapped = false;
        var swap = state.SwapModelAsync(() => { swapped = true; return Task.CompletedTask; }, default);
        Assert.False(swapped);
        leaveFirst.SetResult();
        await first;
        Assert.False(swapped);
        leaveSecond.SetResult();
        await Task.WhenAll(second, swap).WaitAsync(ContinuousBatchSchedulerTests.Timeout);
        Assert.True(swapped);
    }

    [Fact]
    public async Task HttpRequestsShareBatchesAndBothEndpointsKeepResponseOwnership()
    {
        using var model = new CountingModel();
        using var pages = new PagedKvCacheFactory(1, 1, 2, 2, 128);
        using var state = CreateState(model, pages);
        await using var app = ServerStartup.BuildApp(state, []);
        app.Urls.Add("http://127.0.0.1:0");
        await app.StartAsync();
        var server = app.Services.GetRequiredService<IServer>();
        string address = server.Features.Get<IServerAddressesFeature>()!.Addresses.Single();
        using var client = new HttpClient { BaseAddress = new Uri(address), Timeout = ContinuousBatchSchedulerTests.Timeout };
        using var releaseForward = new ManualResetEventSlim();
        var entered = Signal();
        model.OnCall = call =>
        {
            if (call != 1) return;
            entered.TrySetResult();
            if (!releaseForward.Wait(ContinuousBatchSchedulerTests.Timeout)) throw new TimeoutException("Test forward gate");
        };
        var first = client.PostAsJsonAsync("/v1/completions", new { prompt = "a", max_tokens = 6, temperature = 0 });
        await entered.Task.WaitAsync(ContinuousBatchSchedulerTests.Timeout);
        var second = client.PostAsJsonAsync("/v1/completions", new { prompt = "k", max_tokens = 3, temperature = 0, stream = true });
        try
        {
            using var deadline = new CancellationTokenSource(ContinuousBatchSchedulerTests.Timeout);
            while (state.Scheduler!.GetMetrics().QueueDepth == 0) await Task.Delay(5, deadline.Token);
        }
        finally { releaseForward.Set(); }
        using var firstResponse = await first;
        firstResponse.EnsureSuccessStatusCode();
        using var json = JsonDocument.Parse(await firstResponse.Content.ReadAsStringAsync());
        Assert.Equal("bcdefg", json.RootElement.GetProperty("choices")[0].GetProperty("text").GetString());
        using var secondResponse = await second;
        secondResponse.EnsureSuccessStatusCode();
        string events = await secondResponse.Content.ReadAsStringAsync();
        string text = string.Concat(events.Split('\n').Where(l => l.StartsWith("data: {")).Select(l =>
        {
            using var item = JsonDocument.Parse(l[6..]);
            return item.RootElement.GetProperty("choices")[0].GetProperty("text").GetString();
        }));
        Assert.Equal("lmn", text);
        Assert.Contains("data: [DONE]", events);
        Assert.Contains(model.Batches, b => b.Length == 2);

        using var chat = await client.PostAsJsonAsync("/v1/chat/completions",
            new { messages = new[] { new { role = "user", content = "a" } }, max_tokens = 2, temperature = 0 });
        chat.EnsureSuccessStatusCode();
        using var chatJson = JsonDocument.Parse(await chat.Content.ReadAsStringAsync());
        Assert.Equal("bc", chatJson.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        // A withheld stop-prefix token has no text yet, but must still carry its logprob.
        using var streamedChat = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            messages = new[] { new { role = "user", content = "a" } }, max_tokens = 5, temperature = 0,
            stream = true, logprobs = true, stop = new[] { "cd" }
        });
        streamedChat.EnsureSuccessStatusCode();
        string chatEvents = await streamedChat.Content.ReadAsStringAsync();
        int logprobCount = 0;
        var chatText = new System.Text.StringBuilder();
        foreach (string line in chatEvents.Split('\n').Where(l => l.StartsWith("data: {")))
        {
            using var item = JsonDocument.Parse(line[6..]);
            var choices = item.RootElement.GetProperty("choices");
            if (choices.GetArrayLength() == 0) continue;
            var choice = choices[0];
            if (choice.TryGetProperty("logprobs", out var logprobs) && logprobs.ValueKind == JsonValueKind.Object)
                logprobCount += logprobs.GetProperty("content").GetArrayLength();
            if (choice.GetProperty("delta").TryGetProperty("content", out var content)) chatText.Append(content.GetString());
        }
        Assert.Equal("b", chatText.ToString());
        Assert.Equal(3, logprobCount);
        Assert.Equal(pages.Pool.TotalBlocks, pages.Pool.FreeBlocks);
        await app.StopAsync();
    }

    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static ServerState CreateState(CountingModel model, PagedKvCacheFactory pages)
    {
        var tokenizer = new AlphabetTokenizer();
        var scheduler = new ContinuousBatchScheduler(model, tokenizer, pages,
            new() { MaxSequences = 3, MaxBatchedTokens = 8, PrefillChunkSize = 3 });
        _ = scheduler.RunLoopAsync(CancellationToken.None);
        return new ServerState
        {
            Options = new() { Model = "test", ContinuousBatching = true },
            Model = model, Config = model.Config, Tokenizer = tokenizer, Scheduler = scheduler,
            Generator = new TextGenerator(model, tokenizer), IsReady = true,
            ChatTemplate = new JinjaChatTemplate("a", "", "")
        };
    }
}
