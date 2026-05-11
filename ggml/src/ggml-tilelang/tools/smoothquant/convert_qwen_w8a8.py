#!/usr/bin/env python3
"""Convenience wrapper for Qwen SmoothQuant W8A8 sidecar generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, type=Path, help="Qwen HF safetensors directory")
    parser.add_argument("--gguf", required=True, type=Path, help="runtime GGUF used for tensor name/shape validation")
    parser.add_argument("--out", required=True, type=Path, help="output .tlw8a8 sidecar")
    parser.add_argument("--calib", type=Path, help="plain text, JSONL, or JSON calibration prompts")
    parser.add_argument("--act-amax", type=Path, help="existing act_amax .npz; skips calibration when present")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--smooth-min", type=float, default=1.0e-4)
    parser.add_argument("--smooth-max", type=float, default=1.0e4)
    parser.add_argument("--smooth-group", default="", help="comma-separated shared SmoothQuant groups: qkv,gate_up")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-regex", help="override default packer tensor selection")
    parser.add_argument("--exclude-regex")
    parser.add_argument("--check-samples", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    act_amax = args.act_amax
    if act_amax is None:
        act_amax = args.out.with_suffix(".act-amax.npz")

    if not act_amax.exists():
        calib_cmd = [
            sys.executable,
            str(TOOL_DIR / "calibrate_smoothquant.py"),
            "--model", str(args.hf_dir),
            "--output", str(act_amax),
            "--max-samples", str(args.max_samples),
            "--seq-len", str(args.seq_len),
            "--device", args.device,
            "--dtype", args.dtype,
        ]
        if args.calib is not None:
            calib_cmd += ["--prompts", str(args.calib)]
        if args.trust_remote_code:
            calib_cmd += ["--trust-remote-code"]
        run(calib_cmd, dry_run=args.dry_run)
    else:
        print(f"using existing act_amax: {act_amax}", flush=True)

    pack_cmd = [
        sys.executable,
        str(TOOL_DIR / "pack_w8a8_sidecar.py"),
        "--model", str(args.hf_dir),
        "--gguf", str(args.gguf),
        "--act-amax", str(act_amax),
        "--smooth", "smoothquant",
        "--alpha", str(args.alpha),
        "--smooth-min", str(args.smooth_min),
        "--smooth-max", str(args.smooth_max),
        "--output", str(args.out),
        "--check-samples", str(args.check_samples),
    ]
    if args.smooth_group:
        pack_cmd += ["--smooth-group", args.smooth_group]
    if args.include_regex:
        pack_cmd += ["--include-regex", args.include_regex]
    if args.exclude_regex:
        pack_cmd += ["--exclude-regex", args.exclude_regex]
    run(pack_cmd, dry_run=args.dry_run)

    print("sidecar:", args.out)
    print("act_amax:", act_amax)


if __name__ == "__main__":
    main()
