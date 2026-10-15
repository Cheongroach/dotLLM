using BenchmarkDotNet.Attributes;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;

namespace DotLLM.Benchmarks;

/// <summary>Compares complete legacy gather+attention with direct page reads for one decode token.</summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class PagedAttentionBenchmarks
{
    private PagedKvCacheFactory _factory = null!;
    private PagedKvCache _cache = null!;
    private UnmanagedTensor _query = null!, _output = null!;
    private const int Heads = 16, KvHeads = 4, Dim = 64;

    /// <summary>Cached context length.</summary>
    [Params(512, 4096, 16384)]
    public int ContextLength { get; set; }

    /// <summary>Builds a deterministic fragmented cache before measuring.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _factory = new PagedKvCacheFactory(1, KvHeads, Dim, 16, ContextLength + 16);
        _cache = _factory.Create(ContextLength);
        _query = UnmanagedTensor.Allocate(new TensorShape(1, Heads * Dim), DType.Float32);
        _output = UnmanagedTensor.Allocate(_query.Shape, DType.Float32);
        using var data = UnmanagedTensor.Allocate(new TensorShape(ContextLength, KvHeads * Dim), DType.Float32);
        var rng = new Random(123);
        var values = new Span<float>((void*)data.DataPointer, (int)data.ElementCount);
        var q = new Span<float>((void*)_query.DataPointer, (int)_query.ElementCount);
        for (int i = 0; i < values.Length; i++) values[i] = rng.NextSingle() * 2 - 1;
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2 - 1;
        int[] positions = Enumerable.Range(0, ContextLength).ToArray();
        var first = new TensorRef(16, KvHeads * Dim, DType.Float32, -1, data.DataPointer);
        _cache.Update(first, first, positions.AsSpan(0, 16), 0);
        using var blocker = _factory.Create(16);
        blocker.Update(first, first, positions.AsSpan(0, 16), 0);
        var rest = new TensorRef(ContextLength - 16, KvHeads * Dim, DType.Float32, -1, data.DataPointer + 16 * KvHeads * Dim * sizeof(float));
        _cache.Update(rest, rest, positions.AsSpan(16), 0);
        // Allocate legacy staging in setup, not the timed region.
        _cache.GetKeysRef(0);
        _cache.GetValuesRef(0);
    }

    /// <summary>Historical path: gather both KV tensors and call contiguous attention.</summary>
    [Benchmark(Baseline = true)]
    public void GatherAndAttend()
    {
        var k = _cache.GetKeysRef(0);
        var v = _cache.GetValuesRef(0);
        Attention.Execute((float*)_query.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer,
            (float*)_output.DataPointer, 1, ContextLength, Heads, KvHeads, Dim, ContextLength - 1, null);
    }

    /// <summary>Direct physical page lookup with online softmax.</summary>
    [Benchmark]
    public void AttendToPages() => PagedAttention.Execute((float*)_query.DataPointer, _cache, 0,
        (float*)_output.DataPointer, 1, Heads, KvHeads, Dim, ContextLength - 1);

    /// <summary>Releases all host storage.</summary>
    [GlobalCleanup]
    public void Cleanup() { _cache.Dispose(); _factory.Dispose(); _query.Dispose(); _output.Dispose(); }
}
