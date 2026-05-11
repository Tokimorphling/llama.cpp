#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from registry import SUPPORTED_KERNELS, build_all, parse_kernel_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AOT CUDA kernels for ggml-tilelang.")
    parser.add_argument("--target", default="cuda", choices=["cuda"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", default=",".join(SUPPORTED_KERNELS),
                        help=f"comma-separated kernels to generate: {', '.join(SUPPORTED_KERNELS)}")
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--w8a8-gemm-block-m", type=int, default=96)
    parser.add_argument("--w8a8-gemm-block-n", type=int, default=128)
    parser.add_argument("--w8a8-gemm-block-k", type=int, default=64)
    parser.add_argument("--w8a8-gemm-narrow-block-n", type=int, default=64,
                        help="N tile for W8A8 GEMM shapes with N <= 1024")
    parser.add_argument("--w8a8-gemm-threads", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all(
        target=args.target,
        out_dir=args.out,
        block_n=args.block_n,
        kernels=parse_kernel_list(args.kernels),
        command=sys.argv,
        w8a8_gemm_block_m=args.w8a8_gemm_block_m,
        w8a8_gemm_block_n=args.w8a8_gemm_block_n,
        w8a8_gemm_block_k=args.w8a8_gemm_block_k,
        w8a8_gemm_narrow_block_n=args.w8a8_gemm_narrow_block_n,
        w8a8_gemm_threads=args.w8a8_gemm_threads,
    )


if __name__ == "__main__":
    main()
