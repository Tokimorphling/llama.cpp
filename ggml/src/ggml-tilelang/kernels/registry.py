from pathlib import Path

from common import GENERATED_KERNEL_NAMES, lower_cuda_source, provenance_header
from ops.w8a8_gemm import (
    w8a8_gate_up_gemm_dynamic_smem_bytes,
    w8a8_gate_up_silu_scaled_gemm_dot_partial_max,
    w8a8_gemm_dot,
    w8a8_gemm_dynamic_smem_bytes,
    w8a8_gemm_launcher_source,
    w8a8_quant_from_partial_max_rows,
    w8a8_quant_x_rows,
    w8a8_silu_mul_quant_rows,
)
from ops.w8a8_gemv import (
    w8a8_dot,
    w8a8_launcher_source,
    w8a8_quant_x,
)


SUPPORTED_KERNELS = ("w8a8_gemv", "w8a8_gemm")


def build_all(
        target: str,
        out_dir: Path,
        block_n: int,
        kernels: list[str],
        command: list[str],
        w8a8_gemm_block_m: int,
        w8a8_gemm_block_n: int,
        w8a8_gemm_block_k: int,
        w8a8_gemm_narrow_block_n: int,
        w8a8_gemm_threads: int) -> None:
    if target != "cuda":
        raise ValueError(f"unsupported target: {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    header = provenance_header(target=target, command=command)

    if "w8a8_gemv" in kernels:
        quant_source = lower_cuda_source(
            w8a8_quant_x(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_quant_x"],
            header=header,
        )
        dot_source = lower_cuda_source(
            w8a8_dot(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_dot"],
            header=header,
        )
        (out_dir / "tilelang_w8a8_gemv.cu").write_text(quant_source + "\n" + dot_source, encoding="utf-8")
        (out_dir / "tilelang_w8a8_gemv_launcher.cu").write_text(w8a8_launcher_source(header), encoding="utf-8")

    if "w8a8_gemm" in kernels:
        quant_rows_source = lower_cuda_source(
            w8a8_quant_x_rows(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_quant_x_rows"],
            header=header,
        )
        quant_from_partial_max_rows_source = lower_cuda_source(
            w8a8_quant_from_partial_max_rows(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_quant_from_partial_max_rows"],
            header=header,
        )
        silu_mul_quant_rows_source = lower_cuda_source(
            w8a8_silu_mul_quant_rows(block_n=block_n),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_silu_mul_quant_rows"],
            header=header,
        )
        gemm_dot_source = lower_cuda_source(
            w8a8_gemm_dot(
                block_m=w8a8_gemm_block_m,
                block_n=w8a8_gemm_block_n,
                block_k=w8a8_gemm_block_k,
                threads=w8a8_gemm_threads),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_gemm_dot"],
            header=header,
            keep_tl_templates=True,
        )
        narrow_gemm_dot_source = lower_cuda_source(
            w8a8_gemm_dot(
                block_m=w8a8_gemm_block_m,
                block_n=w8a8_gemm_narrow_block_n,
                block_k=w8a8_gemm_block_k,
                threads=w8a8_gemm_threads),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_gemm_dot_narrow_n"],
            header=header,
            keep_tl_templates=True,
        )
        gate_up_block_m = w8a8_gemm_block_m
        gate_up_block_n = w8a8_gemm_narrow_block_n
        gate_up_silu_scaled_gemm_dot_partial_max_source = lower_cuda_source(
            w8a8_gate_up_silu_scaled_gemm_dot_partial_max(
                block_m=gate_up_block_m,
                block_n=gate_up_block_n,
                block_k=w8a8_gemm_block_k,
                threads=w8a8_gemm_threads),
            target=target,
            kernel_name=GENERATED_KERNEL_NAMES["w8a8_gate_up_silu_scaled_gemm_dot_partial_max"],
            header=header,
            keep_tl_templates=True,
        )
        (out_dir / "tilelang_w8a8_gemm.cu").write_text(
            "\n".join([
                quant_rows_source,
                quant_from_partial_max_rows_source,
                silu_mul_quant_rows_source,
                gemm_dot_source,
                narrow_gemm_dot_source,
                gate_up_silu_scaled_gemm_dot_partial_max_source,
            ]),
            encoding="utf-8")
        dynamic_smem_bytes = w8a8_gemm_dynamic_smem_bytes(
            w8a8_gemm_block_m,
            w8a8_gemm_block_n,
            w8a8_gemm_block_k)
        narrow_dynamic_smem_bytes = w8a8_gemm_dynamic_smem_bytes(
            w8a8_gemm_block_m,
            w8a8_gemm_narrow_block_n,
            w8a8_gemm_block_k)
        gate_up_dynamic_smem_bytes = w8a8_gate_up_gemm_dynamic_smem_bytes(
            gate_up_block_m,
            gate_up_block_n,
            w8a8_gemm_block_k)
        (out_dir / "tilelang_w8a8_gemm_launcher.cu").write_text(
            w8a8_gemm_launcher_source(
                header,
                block_m=w8a8_gemm_block_m,
                block_n=w8a8_gemm_block_n,
                block_k=w8a8_gemm_block_k,
                threads=w8a8_gemm_threads,
                dynamic_smem_bytes=dynamic_smem_bytes,
                narrow_block_n=w8a8_gemm_narrow_block_n,
                narrow_dynamic_smem_bytes=narrow_dynamic_smem_bytes,
                gate_up_block_m=gate_up_block_m,
                gate_up_block_n=gate_up_block_n,
                gate_up_dynamic_smem_bytes=gate_up_dynamic_smem_bytes),
            encoding="utf-8")

def parse_kernel_list(value: str) -> list[str]:
    kernels = [kernel.strip() for kernel in value.split(",") if kernel.strip()]
    unknown = sorted(set(kernels) - set(SUPPORTED_KERNELS))
    if unknown:
        raise ValueError(f"unsupported kernels: {', '.join(unknown)}")
    if not kernels:
        raise ValueError("at least one kernel must be requested")
    return kernels
