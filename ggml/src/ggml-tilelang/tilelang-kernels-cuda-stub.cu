#include "tilelang-kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

static bool tilelang_cuda_sync_enabled() {
    const char * env = std::getenv("GGML_TILELANG_SYNC");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void tilelang_cuda_debug_sync(cudaStream_t stream) {
    if (!tilelang_cuda_sync_enabled()) {
        return;
    }

    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

// A naive CUDA reference kernel for F16 GEMV (y = W * x)
// ggml tensors are stored such that ne[0] is contiguous.
// For W, ne[0] = K, ne[1] = N. So it is stored as N rows of size K.
// row i of W starts at w_f16 + i * K.
static __global__ void tilelang_f16_gemv_kernel(
    const half * __restrict__ w_f16,
    const float * __restrict__ x_f32,
    float * __restrict__ y_f32,
    int K, int N) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        const half * w_row = w_f16 + row * K;
        for (int i = 0; i < K; ++i) {
            sum += __half2float(w_row[i]) * x_f32[i];
        }
        y_f32[row] = sum;
    }
}

extern "C" void tilelang_f16_gemv(
    const void * w_f16,
    const float * x_f32,
    float * y_f32,
    int K,
    int N,
    void * stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaStream_t cu_stream = (cudaStream_t)stream;
    tilelang_cuda_debug_sync(cu_stream);
    tilelang_f16_gemv_kernel<<<blocks, threads, 0, cu_stream>>>(
        (const half *)w_f16, x_f32, y_f32, K, N
    );
    tilelang_cuda_debug_sync(cu_stream);
}
