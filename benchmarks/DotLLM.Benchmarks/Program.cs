using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using DotLLM.Benchmarks.Columns;

if (args.Length > 0 && args[0] == "--continuous")
{
    await DotLLM.Benchmarks.ContinuousBatchingBenchmark.RunAsync(args[1..]);
    return;
}

var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddColumn(new PrefillTokPerSecColumn())
    .AddColumn(new DecodeTokPerSecColumn());

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
