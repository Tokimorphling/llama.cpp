import textwrap

import tilelang.language as T

from common import GENERATED_KERNEL_NAMES, cuda_sync_helper_source


def w8a8_quant_x_rows(block_n: int = 256):
    M = T.dynamic("M")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        X: T.Tensor((M, K), "float32"),
        Smooth: T.Tensor((K,), "float32"),
        XQ: T.Tensor((M, K), "int8"),
        XScale: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=block_n) as by:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            local_max = T.alloc_local((1,), "float32")
            warp_maxes = T.alloc_shared((8,), "float32", "shared")

            local_max[0] = 0.0
            for k in T.serial(tx, K, block_n):
                v = X[by, k] / Smooth[k]
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
                    XScale[by] = T.max(local_max[0] / 127.0, 1.0e-12)
            T.sync_threads()

            inv_scale = T.alloc_local((1,), "float32")
            inv_scale[0] = 1.0 / XScale[by]
            for k in T.serial(tx, K, block_n):
                qf = T.round((X[by, k] / Smooth[k]) * inv_scale[0])
                q_clamped = T.max(T.min(qf, 127.0), -127.0)
                XQ[by, k] = q_clamped.astype("int8")

    return main

def w8a8_quant_from_partial_max_rows(block_n: int = 256):
    M = T.dynamic("M")
    K = T.dynamic("K")
    P = T.dynamic("P")

    @T.prim_func
    def main(
        XScaled: T.Tensor((M, K), "float32"),
        PartialMax: T.Tensor((M, P), "float32"),
        XQ: T.Tensor((M, K), "int8"),
        XScale: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=block_n) as by:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            local_max = T.alloc_local((1,), "float32")
            warp_maxes = T.alloc_shared((8,), "float32", "shared")

            local_max[0] = 0.0
            for p in T.serial(tx, P, block_n):
                local_max[0] = T.max(local_max[0], PartialMax[by, p])

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
                    XScale[by] = T.max(local_max[0] / 127.0, 1.0e-12)
            T.sync_threads()

            inv_scale = T.alloc_local((1,), "float32")
            inv_scale[0] = 1.0 / XScale[by]
            for k in T.serial(tx, K, block_n):
                qf = T.round(XScaled[by, k] * inv_scale[0])
                q_clamped = T.max(T.min(qf, 127.0), -127.0)
                XQ[by, k] = q_clamped.astype("int8")

    return main

def w8a8_silu_mul_quant_rows(block_n: int = 256):
    M = T.dynamic("M")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        Gate: T.Tensor((M, K), "float32"),
        Up: T.Tensor((M, K), "float32"),
        Smooth: T.Tensor((K,), "float32"),
        XQ: T.Tensor((M, K), "int8"),
        XScale: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=block_n) as by:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            local_max = T.alloc_local((1,), "float32")
            warp_maxes = T.alloc_shared((8,), "float32", "shared")

            local_max[0] = 0.0
            for k in T.serial(tx, K, block_n):
                gate = Gate[by, k]
                v = ((gate * T.sigmoid(gate)) * Up[by, k]) / Smooth[k]
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
                    XScale[by] = T.max(local_max[0] / 127.0, 1.0e-12)
            T.sync_threads()

            inv_scale = T.alloc_local((1,), "float32")
            inv_scale[0] = 1.0 / XScale[by]
            for k in T.serial(tx, K, block_n):
                gate = Gate[by, k]
                v = ((gate * T.sigmoid(gate)) * Up[by, k]) / Smooth[k]
                qf = T.round(v * inv_scale[0])
                q_clamped = T.max(T.min(qf, 127.0), -127.0)
                XQ[by, k] = q_clamped.astype("int8")

    return main

def w8a8_gemm_dot(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        threads: int = 128):
    M = T.dynamic("M")
    K = T.dynamic("K")
    N = T.dynamic("N")

    @T.prim_func
    def main(
        WQ: T.Tensor((N, K), "int8"),
        WScale: T.Tensor((N,), "float32"),
        XQ: T.Tensor((M, K), "int8"),
        XScale: T.Tensor((M,), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_m, block_k), "int8")
            W_shared = T.alloc_shared((block_n, block_k), "int8")
            C_local = T.alloc_fragment((block_m, block_n), "int32")

            T.clear(C_local)

            for ko in T.Pipelined(K // block_k, num_stages=2):
                T.copy(XQ[by * block_m, ko * block_k], X_shared)
                T.copy(WQ[bx * block_n, ko * block_k], W_shared)
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_m, block_n):
                m = by * block_m + i
                n = bx * block_n + j
                if m < M and n < N:
                    Y[m, n] = C_local[i, j].astype("float32") * XScale[m] * WScale[n]

    return main

def w8a8_gate_up_silu_scaled_gemm_dot_partial_max(
        block_m: int = 128,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128):
    M = T.dynamic("M")
    K = T.dynamic("K")
    N = T.dynamic("N")
    NT = T.ceildiv(N, block_n)

    @T.prim_func
    def main(
        WUpQ: T.Tensor((N, K), "int8"),
        WUpScale: T.Tensor((N,), "float32"),
        WGateQ: T.Tensor((N, K), "int8"),
        WGateScale: T.Tensor((N,), "float32"),
        XQ: T.Tensor((M, K), "int8"),
        XScale: T.Tensor((M,), "float32"),
        Smooth: T.Tensor((N,), "float32"),
        YScaled: T.Tensor((M, N), "float32"),
        PartialMax: T.Tensor((M, NT), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_m, block_k), "int8")
            W_up_shared = T.alloc_shared((block_n, block_k), "int8")
            W_gate_shared = T.alloc_shared((block_n, block_k), "int8")
            C_up_local = T.alloc_fragment((block_m, block_n), "int32")
            C_gate_local = T.alloc_fragment((block_m, block_n), "int32")
            H_scaled_local = T.alloc_fragment((block_m, block_n), "float32")
            Amax_local = T.alloc_fragment((block_m,), "float32")

            T.clear(C_up_local)
            T.clear(C_gate_local)

            for ko in T.Pipelined(K // block_k, num_stages=2):
                T.copy(XQ[by * block_m, ko * block_k], X_shared)
                T.copy(WUpQ[bx * block_n, ko * block_k], W_up_shared)
                T.copy(WGateQ[bx * block_n, ko * block_k], W_gate_shared)
                T.gemm(X_shared, W_up_shared, C_up_local, transpose_B=True)
                T.gemm(X_shared, W_gate_shared, C_gate_local, transpose_B=True)

            for i, j in T.Parallel(block_m, block_n):
                m = by * block_m + i
                n = bx * block_n + j
                if m < M and n < N:
                    up = C_up_local[i, j].astype("float32") * XScale[m] * WUpScale[n]
                    gate = C_gate_local[i, j].astype("float32") * XScale[m] * WGateScale[n]
                    scaled = ((gate * T.sigmoid(gate)) * up) / Smooth[n]
                    H_scaled_local[i, j] = scaled
                    YScaled[m, n] = scaled
                else:
                    H_scaled_local[i, j] = 0.0

            T.reduce_absmax(H_scaled_local, Amax_local, dim=1, clear=True)
            for i in T.Parallel(block_m):
                m = by * block_m + i
                if m < M:
                    PartialMax[m, bx] = Amax_local[i]

    return main

def w8a8_gemm_dynamic_smem_bytes(block_m: int, block_n: int, block_k: int, num_stages: int = 2) -> int:
    payload_bytes = num_stages * (block_m + block_n) * block_k
    overhead_bytes = 12 * 1024
    return ((payload_bytes + overhead_bytes + 255) // 256) * 256

def w8a8_gate_up_gemm_dynamic_smem_bytes(block_m: int, block_n: int, block_k: int, num_stages: int = 2) -> int:
    payload_bytes = num_stages * (block_m + 2 * block_n) * block_k
    overhead_bytes = 16 * 1024
    return ((payload_bytes + overhead_bytes + 255) // 256) * 256

def w8a8_gemm_launcher_source(
        header: str,
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        dynamic_smem_bytes: int,
        narrow_block_n: int,
        narrow_dynamic_smem_bytes: int,
        gate_up_block_m: int,
        gate_up_block_n: int,
        gate_up_dynamic_smem_bytes: int) -> str:
    quant_rows_kernel_name = GENERATED_KERNEL_NAMES["w8a8_quant_x_rows"]
    quant_from_partial_max_rows_kernel_name = GENERATED_KERNEL_NAMES["w8a8_quant_from_partial_max_rows"]
    silu_mul_quant_rows_kernel_name = GENERATED_KERNEL_NAMES["w8a8_silu_mul_quant_rows"]
    gemm_dot_kernel_name = GENERATED_KERNEL_NAMES["w8a8_gemm_dot"]
    narrow_gemm_dot_kernel_name = GENERATED_KERNEL_NAMES["w8a8_gemm_dot_narrow_n"]
    gate_up_silu_scaled_gemm_dot_partial_max_kernel_name = GENERATED_KERNEL_NAMES[
        "w8a8_gate_up_silu_scaled_gemm_dot_partial_max"]
    return header + textwrap.dedent(
        f"""\
        // This file owns the stable C ABI for batched W8A8 GEMM experiments.
        #include "../tilelang-kernels.h"

        #include <cuda_runtime.h>
        #include <stdint.h>
        #include <cstdlib>

        {cuda_sync_helper_source()}

        extern "C" __global__ void {quant_rows_kernel_name}(
            const float * __restrict__ smooth_scale,
            const float * __restrict__ x_f32,
            int8_t * __restrict__ x_q,
            float * __restrict__ x_scale,
            int K,
            int M);

        extern "C" __global__ void {quant_from_partial_max_rows_kernel_name}(
            const float * __restrict__ partial_max_f32,
            int8_t * __restrict__ x_q,
            float * __restrict__ x_scale,
            const float * __restrict__ x_scaled_f32,
            int K,
            int M,
            int P);

        extern "C" __global__ void {silu_mul_quant_rows_kernel_name}(
            const float * __restrict__ gate_f32,
            const float * __restrict__ smooth_scale,
            const float * __restrict__ up_f32,
            int8_t * __restrict__ x_q,
            float * __restrict__ x_scale,
            int K,
            int M);

        extern "C" __global__ void {gemm_dot_kernel_name}(
            const int8_t * __restrict__ w_q,
            const float * __restrict__ w_scale,
            const int8_t * __restrict__ x_q,
            const float * __restrict__ x_scale,
            float * __restrict__ y_f32,
            int K,
            int M,
            int N);

        extern "C" __global__ void {narrow_gemm_dot_kernel_name}(
            const int8_t * __restrict__ w_q,
            const float * __restrict__ w_scale,
            const int8_t * __restrict__ x_q,
            const float * __restrict__ x_scale,
            float * __restrict__ y_f32,
            int K,
            int M,
            int N);

        extern "C" __global__ void {gate_up_silu_scaled_gemm_dot_partial_max_kernel_name}(
            float * __restrict__ partial_max_f32,
            const float * __restrict__ smooth_scale,
            const int8_t * __restrict__ w_gate_q,
            const float * __restrict__ w_gate_scale,
            const int8_t * __restrict__ w_up_q,
            const float * __restrict__ w_up_scale,
            const int8_t * __restrict__ x_q,
            const float * __restrict__ x_scale,
            float * __restrict__ y_scaled_f32,
            int K,
            int M,
            int N);

        extern "C" int tilelang_w8a8_gate_up_block_n(void) {{
            return {gate_up_block_n};
        }}

        extern "C" void tilelang_w8a8_quant_x_rows(
            const float * x_f32,
            const float * smooth_scale,
            int8_t * x_q,
            float * x_scale,
            int M,
            int K,
            void * stream
        ) {{
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {quant_rows_kernel_name}<<<M, 256, 0, cu_stream>>>(smooth_scale, x_f32, x_q, x_scale, K, M);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_quant_from_partial_max_rows(
            const float * x_scaled_f32,
            const float * partial_max_f32,
            int8_t * x_q,
            float * x_scale,
            int M,
            int K,
            int P,
            void * stream
        ) {{
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {quant_from_partial_max_rows_kernel_name}<<<M, 256, 0, cu_stream>>>(
                partial_max_f32, x_q, x_scale, x_scaled_f32, K, M, P);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_silu_mul_quant_rows(
            const float * gate_f32,
            const float * up_f32,
            const float * smooth_scale,
            int8_t * x_q,
            float * x_scale,
            int M,
            int K,
            void * stream
        ) {{
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {silu_mul_quant_rows_kernel_name}<<<M, 256, 0, cu_stream>>>(
                gate_f32, smooth_scale, up_f32, x_q, x_scale, K, M);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_gemm_dot(
            const int8_t * w_q,
            const float * w_scale,
            const int8_t * x_q,
            const float * x_scale,
            float * y_f32,
            int M,
            int K,
            int N,
            void * stream
        ) {{
            constexpr int threads = {threads};
            constexpr int block_m = {block_m};
            constexpr int block_n = {block_n};
            constexpr int narrow_block_n = {narrow_block_n};
            constexpr int dynamic_smem_bytes = {dynamic_smem_bytes};
            constexpr int narrow_dynamic_smem_bytes = {narrow_dynamic_smem_bytes};
            const bool use_narrow_n_kernel = N <= 1024;
            const int dispatch_block_n = use_narrow_n_kernel ? narrow_block_n : block_n;
            const int blocks_x = (N + dispatch_block_n - 1) / dispatch_block_n;
            const int blocks_y = (M + block_m - 1) / block_m;
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            if (use_narrow_n_kernel) {{
                {narrow_gemm_dot_kernel_name}<<<dim3(blocks_x, blocks_y), threads, narrow_dynamic_smem_bytes, cu_stream>>>(
                    w_q, w_scale, x_q, x_scale, y_f32, K, M, N);
            }} else {{
                {gemm_dot_kernel_name}<<<dim3(blocks_x, blocks_y), threads, dynamic_smem_bytes, cu_stream>>>(
                    w_q, w_scale, x_q, x_scale, y_f32, K, M, N);
            }}
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_gate_up_silu_scaled_gemm_dot_partial_max(
            const int8_t * w_up_q,
            const float * w_up_scale,
            const int8_t * w_gate_q,
            const float * w_gate_scale,
            const int8_t * x_q,
            const float * x_scale,
            const float * smooth_scale,
            float * y_scaled_f32,
            float * partial_max_f32,
            int M,
            int K,
            int N,
            void * stream
        ) {{
            constexpr int threads = {threads};
            constexpr int block_m = {gate_up_block_m};
            constexpr int block_n = {gate_up_block_n};
            constexpr int dynamic_smem_bytes = {gate_up_dynamic_smem_bytes};
            const int blocks_x = (N + block_n - 1) / block_n;
            const int blocks_y = (M + block_m - 1) / block_m;
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {gate_up_silu_scaled_gemm_dot_partial_max_kernel_name}<<<dim3(blocks_x, blocks_y), threads, dynamic_smem_bytes, cu_stream>>>(
                partial_max_f32, smooth_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale,
                x_q, x_scale, y_scaled_f32, K, M, N);
            tilelang_cuda_debug_sync(cu_stream);
        }}

        extern "C" void tilelang_w8a8_gemm(
            const int8_t * w_q,
            const float * w_scale,
            const float * smooth_scale,
            const float * x_f32,
            float * y_f32,
            int8_t * x_q_scratch,
            float * x_scale_scratch,
            int M,
            int K,
            int N,
            void * stream
        ) {{
            tilelang_w8a8_quant_x_rows(x_f32, smooth_scale, x_q_scratch, x_scale_scratch, M, K, stream);
            tilelang_w8a8_gemm_dot(w_q, w_scale, x_q_scratch, x_scale_scratch, y_f32, M, K, N, stream);
        }}
        """
    )
