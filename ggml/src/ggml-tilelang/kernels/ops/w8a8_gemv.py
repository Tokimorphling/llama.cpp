import textwrap

import tilelang.language as T

from common import GENERATED_KERNEL_NAMES, cuda_sync_helper_source


def w8a8_quant_x(block_n: int = 256):
    K = T.dynamic("K")

    @T.prim_func
    def main(
        X: T.Tensor((K,), "float32"),
        Smooth: T.Tensor((K,), "float32"),
        XQ: T.Tensor((K,), "int8"),
        XScale: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=block_n):
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            local_max = T.alloc_local((1,), "float32")
            warp_maxes = T.alloc_shared((8,), "float32", "shared")

            local_max[0] = 0.0
            for k in T.serial(tx, K, block_n):
                v = X[k] / Smooth[k]
                av = T.if_then_else(v < 0.0, -v, v)
                local_max[0] = T.max(local_max[0], av)

            local_max[0] = T.warp_reduce_max(local_max[0])
            if lane == 0:
                warp_maxes[warp] = local_max[0]
            T.sync_threads()

            if tx < 8:
                local_max[0] = warp_maxes[tx]
            else:
                local_max[0] = 0.0

            if warp == 0:
                local_max[0] = T.warp_reduce_max(local_max[0])
                if tx == 0:
                    XScale[0] = T.max(local_max[0] / 127.0, 1.0e-12)
            T.sync_threads()

            inv_scale = T.alloc_local((1,), "float32")
            inv_scale[0] = 1.0 / XScale[0]
            for k in T.serial(tx, K, block_n):
                qf = T.round((X[k] / Smooth[k]) * inv_scale[0])
                q_clamped = T.max(T.min(qf, 127.0), -127.0)
                XQ[k] = q_clamped.astype("int8")

    return main

def w8a8_dot(block_n: int = 256):
    K = T.dynamic("K")
    N = T.dynamic("N")
    rows_per_block = block_n // 32

    @T.prim_func
    def main(
        WQ: T.Tensor((N, K), "int8"),
        WScale: T.Tensor((N,), "float32"),
        XQ: T.Tensor((K,), "int8"),
        XScale: T.Tensor((1,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, rows_per_block), threads=block_n) as bx:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            n = bx * rows_per_block + warp
            acc = T.alloc_local((1,), "int32")

            acc[0] = 0
            if n < N:
                for k in T.serial(lane, K, 32):
                    acc[0] += (WQ[n, k].astype("int32") * XQ[k].astype("int32"))

                acc[0] = T.warp_reduce_sum(acc[0])
                if lane == 0:
                    Y[n] = acc[0].astype("float32") * XScale[0] * WScale[n]

    return main

def w8a8_launcher_source(header: str) -> str:
    quant_kernel_name = GENERATED_KERNEL_NAMES["w8a8_quant_x"]
    dot_kernel_name = GENERATED_KERNEL_NAMES["w8a8_dot"]
    return header + textwrap.dedent(
        f"""\
        // This file owns the stable C ABI used by ggml-cuda TileLang injection.
        #include "../tilelang-kernels.h"

        #include <cuda_runtime.h>
        #include <stdint.h>

        {cuda_sync_helper_source()}

        extern "C" __global__ void {quant_kernel_name}(
            const float * __restrict__ smooth_scale,
            const float * __restrict__ x_f32,
            int8_t * __restrict__ x_q,
            float * __restrict__ x_scale,
            int K);

        extern "C" __global__ void {dot_kernel_name}(
            const int8_t * __restrict__ w_q,
            const float * __restrict__ w_scale,
            const int8_t * __restrict__ x_q,
            const float * __restrict__ x_scale,
            float * __restrict__ y_f32,
            int K,
            int N);

        extern "C" void tilelang_w8a8_quant_x(
            const float * x_f32,
            const float * smooth_scale,
            int8_t * x_q,
            float * x_scale,
            int K,
            void * stream
        ) {{
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {quant_kernel_name}<<<1, 256, 0, cu_stream>>>(smooth_scale, x_f32, x_q, x_scale, K);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_dot(
            const int8_t * w_q,
            const float * w_scale,
            const int8_t * x_q,
            const float * x_scale,
            float * y_f32,
            int K,
            int N,
            void * stream
        ) {{
            constexpr int threads = 256;
            constexpr int rows_per_block = threads / 32;
            const int blocks = (N + rows_per_block - 1) / rows_per_block;
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {dot_kernel_name}<<<blocks, threads, 0, cu_stream>>>(w_q, w_scale, x_q, x_scale, y_f32, K, N);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_gemv(
            const int8_t * w_q,
            const float * w_scale,
            const float * smooth_scale,
            const float * x_f32,
            float * y_f32,
            int8_t * x_q_scratch,
            float * x_scale_scratch,
            int K,
            int N,
            void * stream
        ) {{
            tilelang_w8a8_quant_x(x_f32, smooth_scale, x_q_scratch, x_scale_scratch, K, stream);
            tilelang_w8a8_dot(w_q, w_scale, x_q_scratch, x_scale_scratch, y_f32, K, N, stream);
        }}
        """
    )
