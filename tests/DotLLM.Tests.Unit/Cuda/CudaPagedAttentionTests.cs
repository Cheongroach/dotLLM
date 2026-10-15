using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public sealed class CudaPagedAttentionTests
{
    [SkippableTheory]
    [InlineData(4, 4, 19, 19, 0)]
    [InlineData(8, 2, 277, 3, 0)]
    [InlineData(4, 1, 277, 1, 17)]
    [InlineData(4, 2, 19, 3, 1)]
    public unsafe void FragmentedPagesMatchScalarAndRecycle(int heads, int kvHeads, int length, int rows, int window)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(Path.Combine(AppContext.BaseDirectory, "ptx"));
        const int dim = 32, pageSize = 4;
        int kvStride = kvHeads * dim, qStride = heads * dim;
        using var pool = new CudaPagedKvCacheFactory(context, 1, kvHeads, dim, pageSize, length + 16);
        using var cache = pool.Create(length);
        using var blocker = pool.Create(pageSize);
        cache.Prepare(pageSize);
        blocker.Prepare(pageSize); // Force the target's later pages to be physically non-contiguous.
        cache.Prepare(length);
        Assert.Throws<NotSupportedException>(() => cache.GetKeysRef(0));

        using var hostK = UnmanagedTensor.Allocate(new TensorShape(length, kvStride), DType.Float16);
        using var hostV = UnmanagedTensor.Allocate(new TensorShape(length, kvStride), DType.Float16);
        using var hostQ = UnmanagedTensor.Allocate(new TensorShape(rows, qStride), DType.Float16);
        using var hostOut = UnmanagedTensor.Allocate(new TensorShape(rows, qStride), DType.Float16);
        var k = new Span<Half>((void*)hostK.DataPointer, length * kvStride);
        var v = new Span<Half>((void*)hostV.DataPointer, length * kvStride);
        var q = new Span<Half>((void*)hostQ.DataPointer, rows * qStride);
        var rng = new Random(123);
        for (int i = 0; i < k.Length; i++) { k[i] = (Half)(rng.NextSingle() * 2 - 1); v[i] = (Half)(rng.NextSingle() * 2 - 1); }
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(rng.NextSingle() * 2 - 1);
        using var deviceK = CudaTensor.Allocate(hostK.Shape, DType.Float16, 0);
        using var deviceV = CudaTensor.Allocate(hostV.Shape, DType.Float16, 0);
        using var deviceQ = CudaTensor.Allocate(hostQ.Shape, DType.Float16, 0);
        using var deviceOut = CudaTensor.Allocate(hostOut.Shape, DType.Float16, 0);
        CudaDriverApi.cuMemcpyHtoD_v2(deviceK.DataPointer, hostK.DataPointer, (nuint)hostK.ByteCount).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD_v2(deviceV.DataPointer, hostV.DataPointer, (nuint)hostV.ByteCount).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD_v2(deviceQ.DataPointer, hostQ.DataPointer, (nuint)hostQ.ByteCount).ThrowOnError();
        kernels.LaunchScatterPagedKv(deviceK.DataPointer, deviceV.DataPointer, pool.Keys(0), pool.Values(0),
            cache.TableDevice, pageSize, length, kvStride, 0, stream.Handle);
        kernels.LaunchPagedAttention(deviceQ.DataPointer, pool.Keys(0), pool.Values(0), deviceOut.DataPointer,
            cache.TableDevice, pageSize, rows, length, heads, kvHeads, dim, length - rows, window, stream.Handle);
        stream.Synchronize();
        CudaDriverApi.cuMemcpyDtoH_v2(hostOut.DataPointer, deviceOut.DataPointer, (nuint)hostOut.ByteCount).ThrowOnError();
        var actual = new ReadOnlySpan<Half>((void*)hostOut.DataPointer, rows * qStride);
        for (int row = 0; row < rows; row++)
        for (int head = 0; head < heads; head++)
        {
            int end = length - rows + row + 1;
            int start = window > 0 ? Math.Max(0, end - window) : 0;
            var scores = new double[end - start];
            for (int t = start; t < end; t++)
            {
                double dot = 0;
                for (int d = 0; d < dim; d++)
                    dot += (double)q[row * qStride + head * dim + d] * (double)k[t * kvStride + head / (heads / kvHeads) * dim + d];
                scores[t - start] = dot / Math.Sqrt(dim);
            }
            double max = scores.Max();
            for (int t = 0; t < scores.Length; t++) scores[t] = Math.Exp(scores[t] - max);
            double sum = scores.Sum();
            for (int d = 0; d < dim; d++)
            {
                double expected = 0;
                for (int t = start; t < end; t++)
                    expected += scores[t - start] / sum * (double)v[t * kvStride + head / (heads / kvHeads) * dim + d];
                Assert.InRange(Math.Abs(expected - (double)actual[row * qStride + head * dim + d]), 0, 0.001);
            }
        }
        cache.Rollback(3);
        Assert.Equal(pool.TotalBlocks - 2, pool.FreeBlocks);
        cache.Dispose();
        blocker.Dispose();
        Assert.Equal(pool.TotalBlocks, pool.FreeBlocks);
        using var recycled = pool.Create(4);
        recycled.Prepare(4);
        Assert.Equal(pool.TotalBlocks - 1, pool.FreeBlocks);
    }
}
