#include "tilelang-kernels.h"

#define GGML_COMMON_DECL_CUDA
#include "../ggml-common.h"

#include <cuda_runtime.h>
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

// A naive CUDA reference kernel for Q8_0 fused dequant GEMV.
// W is stored as N rows of K / QK8_0 blocks, matching ggml block_q8_0 layout.
static __global__ void tilelang_q8_0_gemv_kernel(
    const block_q8_0 * __restrict__ w_q8_0,
    const float * __restrict__ x_f32,
    float * __restrict__ y_f32,
    int K,
    int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        const int blocks_per_row = K / QK8_0;
        const block_q8_0 * w_row = w_q8_0 + row * blocks_per_row;

        for (int kb = 0; kb < blocks_per_row; ++kb) {
            const block_q8_0 block = w_row[kb];
            const float d = __half2float(block.d);
            for (int j = 0; j < QK8_0; ++j) {
                sum += d * (float) block.qs[j] * x_f32[kb * QK8_0 + j];
            }
        }

        y_f32[row] = sum;
    }
}

extern "C" void tilelang_q8_0_gemv(
    const void * w_q8_0,
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
    tilelang_q8_0_gemv_kernel<<<blocks, threads, 0, cu_stream>>>(
        (const block_q8_0 *)w_q8_0, x_f32, y_f32, K, N
    );
    tilelang_cuda_debug_sync(cu_stream);
}
