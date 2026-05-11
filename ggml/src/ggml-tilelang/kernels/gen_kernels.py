#!/usr/bin/env python3
import argparse
import re
import shlex
import sys
import textwrap
from pathlib import Path

import tilelang
import tilelang.language as T


GENERATOR_PATH = "ggml/src/ggml-tilelang/kernels/gen_kernels.py"
GENERATED_KERNEL_NAMES = {
    "f16_gemv": "tilelang_generated_f16_gemv_kernel",
    "q8_0_gemv_thread": "tilelang_generated_q8_0_gemv_thread_kernel",
    "q8_0_gemv_block": "tilelang_generated_q8_0_gemv_block_kernel",
    "q8_0_gemv_warp": "tilelang_generated_q8_0_gemv_warp_kernel",
}
SUPPORTED_KERNELS = ("f16_gemv", "q8_0_gemv")


def f16_gemv(block_n: int = 256):
    K = T.dynamic("K")
    N = T.dynamic("N")

    @T.prim_func
    def main(
        W: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), threads=block_n) as bx:
            tx = T.get_thread_binding(0)
            n = bx * block_n + tx
            acc = T.alloc_local((1,), "float32")

            acc[0] = 0.0
            if n < N:
                for k in T.serial(K):
                    acc[0] += W[n, k].astype("float32") * X[k]
                Y[n] = acc[0]

    return main


def q8_0_gemv_thread(block_n: int = 256):
    K = T.dynamic("K")
    N = T.dynamic("N")
    B = T.dynamic("B")

    @T.prim_func
    def main(
        W: T.Tensor((N, B), "uint8"),
        X: T.Tensor((K,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), threads=block_n) as bx:
            tx = T.get_thread_binding(0)
            n = bx * block_n + tx
            acc = T.alloc_local((1,), "float32")
            d_bits = T.alloc_local((1,), "uint16")

            acc[0] = 0.0
            if n < N:
                for kb in T.serial(K // 32):
                    block_byte = kb * 34
                    d_bits[0] = W[n, block_byte].astype("uint16") | (W[n, block_byte + 1].astype("uint16") << 8)
                    d = T.reinterpret(d_bits[0], "float16").astype("float32")

                    for j in T.serial(32):
                        q_u = W[n, block_byte + 2 + j].astype("int32")
                        q = T.if_then_else(q_u >= 128, q_u - 256, q_u).astype("float32")
                        acc[0] += d * q * X[kb * 32 + j]

                Y[n] = acc[0]

    return main


def q8_0_gemv_block(block_n: int = 256):
    K = T.dynamic("K")
    N = T.dynamic("N")
    B = T.dynamic("B")

    @T.prim_func
    def main(
        W: T.Tensor((N, B), "uint8"),
        X: T.Tensor((K,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N, threads=block_n) as n:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            acc = T.alloc_local((1,), "float32")
            d_bits = T.alloc_local((1,), "uint16")
            warp_sums = T.alloc_shared((8,), "float32", "shared")

            acc[0] = 0.0
            for kb in T.serial(tx, K // 32, block_n):
                block_byte = kb * 34
                d_bits[0] = W[n, block_byte].astype("uint16") | (W[n, block_byte + 1].astype("uint16") << 8)
                d = T.reinterpret(d_bits[0], "float16").astype("float32")

                for j in T.serial(32):
                    q_u = W[n, block_byte + 2 + j].astype("int32")
                    q = T.if_then_else(q_u >= 128, q_u - 256, q_u).astype("float32")
                    acc[0] += d * q * X[kb * 32 + j]

            acc[0] = T.warp_reduce_sum(acc[0])
            if lane == 0:
                warp_sums[warp] = acc[0]
            T.sync_threads()

            if tx < 8:
                acc[0] = warp_sums[tx]
            else:
                acc[0] = 0.0

            if warp == 0:
                acc[0] = T.warp_reduce_sum(acc[0])
                if tx == 0:
                    Y[n] = acc[0]

    return main


def q8_0_gemv_warp(block_n: int = 256):
    K = T.dynamic("K")
    N = T.dynamic("N")
    B = T.dynamic("B")
    rows_per_block = block_n // 32

    @T.prim_func
    def main(
        W: T.Tensor((N, B), "uint8"),
        X: T.Tensor((K,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, rows_per_block), threads=block_n) as bx:
            tx = T.get_thread_binding(0)
            lane = tx % 32
            warp = tx // 32
            n = bx * rows_per_block + warp
            acc = T.alloc_local((1,), "float32")
            d_bits = T.alloc_local((1,), "uint16")

            acc[0] = 0.0
            if n < N:
                for kb in T.serial(lane, K // 32, 32):
                    block_byte = kb * 34
                    d_bits[0] = W[n, block_byte].astype("uint16") | (W[n, block_byte + 1].astype("uint16") << 8)
                    d = T.reinterpret(d_bits[0], "float16").astype("float32")

                    for j in T.serial(32):
                        q_u = W[n, block_byte + 2 + j].astype("int32")
                        q = T.if_then_else(q_u >= 128, q_u - 256, q_u).astype("float32")
                        acc[0] += d * q * X[kb * 32 + j]

                acc[0] = T.warp_reduce_sum(acc[0])
                if lane == 0:
                    Y[n] = acc[0]

    return main


def provenance_header(target: str, command: list[str]) -> str:
    command_line = " ".join(shlex.quote(arg) for arg in command)
    tilelang_version = getattr(tilelang, "__version__", "unknown")

    return textwrap.dedent(
        f"""\
        // Generated by {GENERATOR_PATH}
        // TileLang version: {tilelang_version}
        // Target: {target}
        // Command: {command_line}
        // Do not edit manually.
        """
    )


def cuda_sync_helper_source() -> str:
    return textwrap.dedent(
        """\
        static bool tilelang_cuda_sync_enabled() {
            const char * env = std::getenv("GGML_TILELANG_SYNC");
            return env != nullptr && env[0] != '\\0' && env[0] != '0';
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

        """
    )


def normalize_cuda_source(source: str, kernel_name: str, header: str) -> str:
    source, count = re.subn(r"\bmain_kernel\b", kernel_name, source)
    if count == 0:
        raise RuntimeError("TileLang CUDA source did not contain main_kernel")

    lines = []
    skip_bf16_block = False
    for line in source.splitlines():
        if line.startswith("#include <tl_templates/"):
            continue
        if line == "#ifdef ENABLE_BF16":
            skip_bf16_block = True
            continue
        if skip_bf16_block:
            if line == "#endif":
                skip_bf16_block = False
            continue
        lines.append(line)
    source = "\n".join(lines).strip() + "\n"

    prefix = header + textwrap.dedent(
        """\
        #include <cuda_fp16.h>
        #include <cuda_runtime.h>
        #include <stdint.h>

        using half_t = half;
        using uchar = unsigned char;
        using ushort = unsigned short;

        #ifndef TILELANG_AOT_WARP_REDUCE_SUM_DEFINED
        #define TILELANG_AOT_WARP_REDUCE_SUM_DEFINED
        namespace tl {
        __device__ __forceinline__ float warp_reduce_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            return value;
        }
        }
        #endif

        """
    )
    return prefix + source


def f16_launcher_source(header: str) -> str:
    kernel_name = GENERATED_KERNEL_NAMES["f16_gemv"]
    return header + textwrap.dedent(
        f"""\
        // This file owns the stable C ABI used by ggml-tilelang.
        #include "../tilelang-kernels.h"

        #include <cuda_fp16.h>
        #include <cuda_runtime.h>
        #include <cstdlib>

        {cuda_sync_helper_source()}

        extern "C" __global__ void {kernel_name}(
            const half * __restrict__ w_f16,
            const float * __restrict__ x_f32,
            float * __restrict__ y_f32,
            int K,
            int N);

        extern "C" void tilelang_f16_gemv(
            const void * w_f16,
            const float * x_f32,
            float * y_f32,
            int K,
            int N,
            void * stream
        ) {{
            constexpr int threads = 256;
            const int blocks = (N + threads - 1) / threads;
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            {kernel_name}<<<blocks, threads, 0, cu_stream>>>(
                static_cast<const half *>(w_f16),
                x_f32,
                y_f32,
                K,
                N
            );
            tilelang_cuda_debug_sync(cu_stream);
        }}
        """
    )


def q8_0_launcher_source(header: str) -> str:
    thread_kernel_name = GENERATED_KERNEL_NAMES["q8_0_gemv_thread"]
    block_kernel_name = GENERATED_KERNEL_NAMES["q8_0_gemv_block"]
    warp_kernel_name = GENERATED_KERNEL_NAMES["q8_0_gemv_warp"]
    return header + textwrap.dedent(
        f"""\
        // This file owns the stable C ABI used by ggml-tilelang.
        #include "../tilelang-kernels.h"

        #include <cuda_fp16.h>
        #include <cuda_runtime.h>
        #include <stdint.h>
        #include <cstdlib>
        #include <cstring>

        {cuda_sync_helper_source()}

        static bool tilelang_q8_0_use_block_variant() {{
            const char * env = std::getenv("GGML_TILELANG_Q8_0_VARIANT");
            return env != nullptr && (std::strcmp(env, "block") == 0 || std::strcmp(env, "row1") == 0);
        }}

        static bool tilelang_q8_0_use_warp_variant() {{
            const char * env = std::getenv("GGML_TILELANG_Q8_0_VARIANT");
            return env != nullptr && std::strcmp(env, "warp") == 0;
        }}

        extern "C" __global__ void {thread_kernel_name}(
            const unsigned char * __restrict__ w_q8_0,
            const float * __restrict__ x_f32,
            float * __restrict__ y_f32,
            int B,
            int K,
            int N);

        extern "C" __global__ void {block_kernel_name}(
            const unsigned char * __restrict__ w_q8_0,
            const float * __restrict__ x_f32,
            float * __restrict__ y_f32,
            int B,
            int K,
            int N);

        extern "C" __global__ void {warp_kernel_name}(
            const unsigned char * __restrict__ w_q8_0,
            const float * __restrict__ x_f32,
            float * __restrict__ y_f32,
            int B,
            int K,
            int N);

        extern "C" void tilelang_q8_0_gemv(
            const void * w_q8_0,
            const float * x_f32,
            float * y_f32,
            int K,
            int N,
            void * stream
        ) {{
            constexpr int threads = 256;
            const int row_bytes = (K / 32) * 34;
            cudaStream_t cu_stream = (cudaStream_t) stream;
            tilelang_cuda_debug_sync(cu_stream);
            if (tilelang_q8_0_use_warp_variant()) {{
                constexpr int rows_per_block = threads / 32;
                const int blocks = (N + rows_per_block - 1) / rows_per_block;
                {warp_kernel_name}<<<blocks, threads, 0, cu_stream>>>(
                    static_cast<const unsigned char *>(w_q8_0),
                    x_f32,
                    y_f32,
                    row_bytes,
                    K,
                    N
                );
            }} else if (tilelang_q8_0_use_block_variant()) {{
                {block_kernel_name}<<<N, threads, 0, cu_stream>>>(
                    static_cast<const unsigned char *>(w_q8_0),
                    x_f32,
                    y_f32,
                    row_bytes,
                    K,
                    N
                );
            }} else {{
                const int blocks = (N + threads - 1) / threads;
                {thread_kernel_name}<<<blocks, threads, 0, cu_stream>>>(
                    static_cast<const unsigned char *>(w_q8_0),
                    x_f32,
                    y_f32,
                    row_bytes,
                    K,
                    N
                );
            }}
            tilelang_cuda_debug_sync(cu_stream);
        }}
        """
    )


def lower_cuda_source(program, target: str, kernel_name: str, header: str) -> str:
    artifact = tilelang.lower(
        program,
        target=target,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    return normalize_cuda_source(artifact.kernel_source, kernel_name=kernel_name, header=header)


def build_all(target: str, out_dir: Path, block_n: int, kernels: list[str], command: list[str]) -> None:
    if target != "cuda":
        raise ValueError(f"unsupported target: {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    header = provenance_header(target=target, command=command)

    if "f16_gemv" in kernels:
        cuda_source = lower_cuda_source(
            f16_gemv(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["f16_gemv"],
            header=header,
        )
        (out_dir / "tilelang_f16_gemv.cu").write_text(cuda_source, encoding="utf-8")
        (out_dir / "tilelang_f16_gemv_launcher.cu").write_text(f16_launcher_source(header), encoding="utf-8")

    if "q8_0_gemv" in kernels:
        thread_source = lower_cuda_source(
            q8_0_gemv_thread(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["q8_0_gemv_thread"],
            header=header,
        )
        block_source = lower_cuda_source(
            q8_0_gemv_block(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["q8_0_gemv_block"],
            header=header,
        )
        warp_source = lower_cuda_source(
            q8_0_gemv_warp(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["q8_0_gemv_warp"],
            header=header,
        )
        (out_dir / "tilelang_q8_0_gemv.cu").write_text(thread_source + "\n" + block_source + "\n" + warp_source, encoding="utf-8")
        (out_dir / "tilelang_q8_0_gemv_launcher.cu").write_text(q8_0_launcher_source(header), encoding="utf-8")


def parse_kernel_list(value: str) -> list[str]:
    kernels = [kernel.strip() for kernel in value.split(",") if kernel.strip()]
    unknown = sorted(set(kernels) - set(SUPPORTED_KERNELS))
    if unknown:
        raise ValueError(f"unsupported kernels: {', '.join(unknown)}")
    if not kernels:
        raise ValueError("at least one kernel must be requested")
    return kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AOT CUDA kernels for ggml-tilelang.")
    parser.add_argument("--target", default="cuda", choices=["cuda"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", default=",".join(SUPPORTED_KERNELS),
                        help=f"comma-separated kernels to generate: {', '.join(SUPPORTED_KERNELS)}")
    parser.add_argument("--block-n", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_all(
        target=args.target,
        out_dir=args.out,
        block_n=args.block_n,
        kernels=parse_kernel_list(args.kernels),
        command=sys.argv,
    )
