#include "tilelang-kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct shape {
    int K;
    int N;
};

struct result {
    double ms;
    double gbps;
    double max_abs_err;
    double max_rel_err;
};

#define CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void q8_0_cuda_ref_warp_kernel(
        const unsigned char * __restrict__ w_q8_0,
        const float * __restrict__ x_f32,
        float * __restrict__ y_f32,
        int row_bytes,
        int K,
        int N) {
    constexpr int rows_per_block = 8;
    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int warp = tx >> 5;
    const int row = blockIdx.x * rows_per_block + warp;

    float acc = 0.0f;
    if (row < N) {
        const unsigned char * w_row = w_q8_0 + (size_t) row * row_bytes;
        for (int kb = lane; kb < K / 32; kb += 32) {
            const int block_byte = kb * 34;
            const unsigned short d_bits =
                (unsigned short) w_row[block_byte] |
                ((unsigned short) w_row[block_byte + 1] << 8);
            const float d = __half2float(__ushort_as_half(d_bits));

            for (int j = 0; j < 32; ++j) {
                const int q = (int) (int8_t) w_row[block_byte + 2 + j];
                acc += d * (float) q * x_f32[kb * 32 + j];
            }
        }
    }

    acc = warp_sum(acc);
    if (row < N && lane == 0) {
        y_f32[row] = acc;
    }
}

void cuda_ref_q8_0_gemv(
        const void * w_q8_0,
        const float * x_f32,
        float * y_f32,
        int K,
        int N,
        void * stream) {
    constexpr int threads = 256;
    constexpr int rows_per_block = threads / 32;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    const int row_bytes = (K / 32) * 34;
    q8_0_cuda_ref_warp_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(
        (const unsigned char *) w_q8_0, x_f32, y_f32, row_bytes, K, N);
}

uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

void fill_inputs(std::vector<unsigned char> & w, std::vector<float> & x, int K, int N) {
    const int row_bytes = (K / 32) * 34;
    std::mt19937 rng((uint32_t) K * 1315423911u + (uint32_t) N);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> d_dist(0.001f, 0.08f);
    std::uniform_int_distribution<int> q_dist(-64, 63);

    for (float & v : x) {
        v = x_dist(rng);
    }

    for (int row = 0; row < N; ++row) {
        unsigned char * w_row = w.data() + (size_t) row * row_bytes;
        for (int kb = 0; kb < K / 32; ++kb) {
            const int block_byte = kb * 34;
            const uint16_t d = half_bits(d_dist(rng));
            w_row[block_byte] = (unsigned char) (d & 0xff);
            w_row[block_byte + 1] = (unsigned char) (d >> 8);
            for (int j = 0; j < 32; ++j) {
                const int8_t q = (int8_t) q_dist(rng);
                w_row[block_byte + 2 + j] = (unsigned char) q;
            }
        }
    }
}

void set_variant(const char * variant) {
#if defined(_WIN32)
    _putenv_s("GGML_TILELANG_Q8_0_VARIANT", variant);
    _putenv_s("GGML_TILELANG_SYNC", "0");
#else
    setenv("GGML_TILELANG_Q8_0_VARIANT", variant, 1);
    setenv("GGML_TILELANG_SYNC", "0", 1);
#endif
}

template <typename Fn>
double time_kernel(cudaStream_t stream, int warmup, int iters, Fn launch) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (double) elapsed_ms / (double) iters;
}

std::pair<double, double> max_errors(const std::vector<float> & ref, const std::vector<float> & got) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double abs_err = std::abs((double) ref[i] - (double) got[i]);
        const double denom = std::max(std::abs((double) ref[i]), 1.0e-6);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, abs_err / denom);
    }
    return {max_abs, max_rel};
}

double logical_gbps(int K, int N, double ms) {
    const size_t row_bytes = (size_t) (K / 32) * 34;
    const size_t logical_bytes =
        (size_t) N * row_bytes +
        (size_t) N * (size_t) K * sizeof(float) +
        (size_t) N * sizeof(float);
    return (double) logical_bytes / (ms * 1.0e6);
}

result run_tilelang_variant(
        const char * variant,
        cudaStream_t stream,
        const void * d_w,
        const float * d_x,
        float * d_y,
        int K,
        int N,
        int warmup,
        int iters,
        const std::vector<float> & ref) {
    set_variant(variant);
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, (size_t) N * sizeof(float), stream));
    const double ms = time_kernel(stream, warmup, iters, [&] {
        tilelang_q8_0_gemv(d_w, d_x, d_y, K, N, stream);
    });

    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpyAsync(got.data(), d_y, (size_t) N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto [max_abs, max_rel] = max_errors(ref, got);
    return {ms, logical_gbps(K, N, ms), max_abs, max_rel};
}

result run_cuda_ref(
        cudaStream_t stream,
        const void * d_w,
        const float * d_x,
        float * d_y,
        int K,
        int N,
        int warmup,
        int iters,
        std::vector<float> & ref) {
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, (size_t) N * sizeof(float), stream));
    const double ms = time_kernel(stream, warmup, iters, [&] {
        cuda_ref_q8_0_gemv(d_w, d_x, d_y, K, N, stream);
    });

    ref.resize(N);
    CUDA_CHECK(cudaMemcpyAsync(ref.data(), d_y, (size_t) N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {ms, logical_gbps(K, N, ms), 0.0, 0.0};
}

shape parse_shape(const char * value) {
    std::string text(value);
    const size_t sep = text.find_first_of("xX,");
    if (sep == std::string::npos) {
        std::fprintf(stderr, "invalid shape '%s', expected KxN\n", value);
        std::exit(2);
    }
    return {std::stoi(text.substr(0, sep)), std::stoi(text.substr(sep + 1))};
}

void usage(const char * argv0) {
    std::printf(
        "usage: %s [--device N] [--warmup N] [--iters N] [--shape KxN]...\n"
        "\n"
        "Measures Q8_0 GEMV [K,N] x [K] -> [N] with CUDA events.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    int device = 0;
    int warmup = 5;
    int iters = 20;
    std::vector<shape> shapes;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            shapes.push_back(parse_shape(argv[++i]));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (shapes.empty()) {
        shapes = {
            {1024, 1024},
            {2048, 2048},
            {4096, 4096},
            {4096, 11008},
            {4096, 32000},
            {8192, 8192},
            {8192, 28672},
        };
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("device,%d,warmup,%d,iters,%d\n", device, warmup, iters);
    std::printf("K,N,variant,time_ms,logical_GBps,max_abs_err,max_rel_err\n");

    for (const shape & s : shapes) {
        if (s.K <= 0 || s.N <= 0 || s.K % 32 != 0) {
            std::fprintf(stderr, "skipping invalid shape K=%d N=%d\n", s.K, s.N);
            continue;
        }

        const size_t row_bytes = (size_t) (s.K / 32) * 34;
        const size_t w_bytes = (size_t) s.N * row_bytes;
        std::vector<unsigned char> h_w(w_bytes);
        std::vector<float> h_x(s.K);
        fill_inputs(h_w, h_x, s.K, s.N);

        void * d_w = nullptr;
        float * d_x = nullptr;
        float * d_y_ref = nullptr;
        float * d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
        CUDA_CHECK(cudaMalloc((void **) &d_x, (size_t) s.K * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &d_y_ref, (size_t) s.N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &d_y, (size_t) s.N * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), (size_t) s.K * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> ref;
        const result cuda_result = run_cuda_ref(stream, d_w, d_x, d_y_ref, s.K, s.N, warmup, iters, ref);
        std::printf("%d,%d,cuda_ref,%.6f,%.3f,%.9g,%.9g\n",
            s.K, s.N, cuda_result.ms, cuda_result.gbps, cuda_result.max_abs_err, cuda_result.max_rel_err);

        for (const char * variant : {"thread", "block", "warp"}) {
            const result tilelang_result = run_tilelang_variant(
                variant, stream, d_w, d_x, d_y, s.K, s.N, warmup, iters, ref);
            std::printf("%d,%d,tilelang_%s,%.6f,%.3f,%.9g,%.9g\n",
                s.K, s.N, variant, tilelang_result.ms, tilelang_result.gbps,
                tilelang_result.max_abs_err, tilelang_result.max_rel_err);
        }

        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y_ref));
        CUDA_CHECK(cudaFree(d_y));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
