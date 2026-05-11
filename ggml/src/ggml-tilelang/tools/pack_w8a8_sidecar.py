#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


MAGIC = b"TLW8A8\x01\x00"
HEADER_STRUCT = struct.Struct("<8sQ")
DATA_ALIGNMENT = 64

DEFAULT_SUFFIXES = (
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def sanitize_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name)


def tensor_is_default_linear(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in DEFAULT_SUFFIXES)


def tensor_to_f32(tensor: Any) -> np.ndarray:
    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float32)

    # BF16 and existing GGUF quantized tensors are exposed as raw bytes by the
    # reader. Dequantize only for explicit prototype packing; production W8A8
    # sidecars should be generated from F16/BF16 source weights.
    return np.asarray(dequantize(tensor.data, qtype), dtype=np.float32)


def stable_sha256(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(data).hexdigest()


def load_act_amax(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}

    if path.suffix == ".npz":
        loaded = np.load(path)
        return {key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files}

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {key: np.asarray(value, dtype=np.float32) for key, value in obj.items()}


def find_act_amax(act_amax: dict[str, np.ndarray], tensor_name: str, K: int) -> tuple[np.ndarray, str]:
    for key in (tensor_name, sanitize_key(tensor_name)):
        if key in act_amax:
            value = np.asarray(act_amax[key], dtype=np.float32).reshape(-1)
            if value.size != K:
                raise ValueError(f"act_amax for {tensor_name} has {value.size} values, expected K={K}")
            return np.maximum(value, 1.0e-8), key
    return np.ones(K, dtype=np.float32), "ones"


def compute_smooth_scale(
        W: np.ndarray,
        act_amax: np.ndarray,
        mode: str,
        alpha: float,
        smooth_min: float,
        smooth_max: float) -> tuple[np.ndarray, str]:
    if mode == "none":
        return np.ones(W.shape[1], dtype=np.float32), "none"

    if mode != "smoothquant":
        raise ValueError(f"unknown smooth mode: {mode}")

    w_amax = np.max(np.abs(W), axis=0).astype(np.float32)
    w_amax = np.maximum(w_amax, 1.0e-8)
    act_amax = np.maximum(act_amax.astype(np.float32), 1.0e-8)

    smooth = np.power(act_amax, alpha) / np.power(w_amax, 1.0 - alpha)
    smooth = np.clip(smooth, smooth_min, smooth_max).astype(np.float32)
    return smooth, "smoothquant"


def quantize_weight(W: np.ndarray, smooth_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    W_smooth = W * smooth_scale.reshape(1, -1)
    row_amax = np.max(np.abs(W_smooth), axis=1).astype(np.float32)
    w_scale = row_amax / np.float32(127.0)
    w_scale = np.where(w_scale == 0.0, np.float32(1.0), w_scale).astype(np.float32)

    W_q = np.rint(W_smooth / w_scale.reshape(-1, 1))
    W_q = np.clip(W_q, -127, 127).astype(np.int8)

    W_deq = W_q.astype(np.float32) * w_scale.reshape(-1, 1)
    abs_err = np.abs(W_deq - W_smooth)
    denom = np.maximum(np.abs(W_smooth), 1.0e-6)
    rel_err = abs_err / denom
    stats = {
        "weight_smooth_amax": float(np.max(np.abs(W_smooth))) if W_smooth.size else 0.0,
        "weight_scale_min": float(np.min(w_scale)) if w_scale.size else 0.0,
        "weight_scale_max": float(np.max(w_scale)) if w_scale.size else 0.0,
        "weight_quant_max_abs_err": float(np.max(abs_err)) if abs_err.size else 0.0,
        "weight_quant_mean_abs_err": float(np.mean(abs_err)) if abs_err.size else 0.0,
        "weight_quant_max_rel_err": float(np.max(rel_err)) if rel_err.size else 0.0,
    }
    return np.ascontiguousarray(W_q), np.ascontiguousarray(w_scale), stats


def select_tensors(reader: GGUFReader, args: argparse.Namespace) -> list[Any]:
    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    selected = []
    for tensor in reader.tensors:
        name = tensor.name
        if include_re is not None:
            if include_re.search(name) is None:
                continue
        elif not tensor_is_default_linear(name):
            if not (args.include_output and name == "output.weight"):
                continue

        if exclude_re is not None and exclude_re.search(name) is not None:
            continue

        if len(tensor.shape) != 2:
            continue
        selected.append(tensor)

    if args.max_tensors is not None:
        selected = selected[:args.max_tensors]
    return selected


def append_blob(blobs: list[bytes], array: np.ndarray) -> dict[str, Any]:
    offset = sum(len(blob) for blob in blobs)
    data = np.ascontiguousarray(array).tobytes(order="C")
    blobs.append(data)
    return {
        "offset": offset,
        "nbytes": len(data),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def write_sidecar(path: Path, manifest: dict[str, Any], blobs: list[bytes]) -> None:
    json_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    data_offset = align_up(HEADER_STRUCT.size + len(json_bytes), DATA_ALIGNMENT)
    manifest["data_offset"] = data_offset

    # The data_offset field changes the JSON length. Iterate until the header is
    # stable; this usually converges in one pass.
    for _ in range(8):
        json_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        new_data_offset = align_up(HEADER_STRUCT.size + len(json_bytes), DATA_ALIGNMENT)
        if new_data_offset == manifest["data_offset"]:
            break
        manifest["data_offset"] = new_data_offset
    else:
        raise RuntimeError("failed to stabilize sidecar header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, len(json_bytes)))
        f.write(json_bytes)
        pad = manifest["data_offset"] - HEADER_STRUCT.size - len(json_bytes)
        f.write(b"\0" * pad)
        for blob in blobs:
            f.write(blob)


def build_sidecar(args: argparse.Namespace) -> None:
    model = Path(args.model)
    output = Path(args.output)
    reader = GGUFReader(model)
    selected = select_tensors(reader, args)
    if args.list:
        for tensor in selected:
            dims = [int(v) for v in tensor.shape.tolist()]
            print(f"{tensor.name}\t{tensor.tensor_type.name}\tK={dims[0]}\tN={dims[1]}")
        return

    if not selected:
        raise SystemExit("no tensors selected")

    act_amax_map = load_act_amax(Path(args.act_amax) if args.act_amax else None)
    smooth_mode = args.smooth
    if smooth_mode == "auto":
        smooth_mode = "smoothquant" if act_amax_map else "none"

    source_stat = model.stat()
    manifest: dict[str, Any] = {
        "format": "ggml-tilelang-w8a8-sidecar",
        "version": 1,
        "data_alignment": DATA_ALIGNMENT,
        "source_model": str(model),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "created_unix": int(time.time()),
        "quantization": {
            "recipe": "smoothquant_rtn_w8a8",
            "smooth": smooth_mode,
            "alpha": args.alpha,
            "activation": "dynamic_per_token_int8",
            "activation_scale": "per_token_float32",
            "weight": "int8",
            "weight_scale": "per_output_channel_float32",
        },
        "layout": {
            "weight_q": "row_major_int8_N_by_K",
            "weight_scale": "float32_N",
            "smooth_scale": "float32_K",
        },
        "tensors": [],
    }
    blobs: list[bytes] = []

    total_wq_bytes = 0
    for idx, tensor in enumerate(selected):
        dims = [int(v) for v in tensor.shape.tolist()]
        K, N = dims[0], dims[1]
        W = tensor_to_f32(tensor)
        if W.shape != (N, K):
            W = W.reshape(N, K)

        act_amax, act_key = find_act_amax(act_amax_map, tensor.name, K)
        smooth_scale, applied_smooth = compute_smooth_scale(
            W, act_amax, smooth_mode, args.alpha, args.smooth_min, args.smooth_max)
        W_q, w_scale, stats = quantize_weight(W, smooth_scale)

        w_q_blob = append_blob(blobs, W_q)
        w_scale_blob = append_blob(blobs, w_scale)
        smooth_blob = append_blob(blobs, smooth_scale.astype(np.float32))
        total_wq_bytes += w_q_blob["nbytes"]

        tensor_entry = {
            "index": idx,
            "name": tensor.name,
            "source_type": tensor.tensor_type.name,
            "source_shape": dims,
            "K": K,
            "N": N,
            "source_sha256": stable_sha256(tensor.data),
            "act_amax_source": act_key,
            "smooth": applied_smooth,
            "blobs": {
                "w_q": w_q_blob,
                "w_scale": w_scale_blob,
                "smooth_scale": smooth_blob,
            },
            "stats": stats,
        }
        manifest["tensors"].append(tensor_entry)

        print(
            f"packed {tensor.name}: type={tensor.tensor_type.name} K={K} N={N} "
            f"smooth={applied_smooth} qerr_max={stats['weight_quant_max_abs_err']:.6g}",
            file=sys.stderr,
        )

    manifest["summary"] = {
        "n_tensors": len(manifest["tensors"]),
        "w_q_bytes": total_wq_bytes,
        "total_blob_bytes": sum(len(blob) for blob in blobs),
    }
    write_sidecar(output, manifest, blobs)
    print(json.dumps({
        "output": str(output),
        "n_tensors": len(manifest["tensors"]),
        "total_blob_bytes": manifest["summary"]["total_blob_bytes"],
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack GGUF linear weights into a TileLang SmoothQuant-style W8A8 sidecar.")
    parser.add_argument("--model", required=True, help="source F16/BF16 GGUF")
    parser.add_argument("--output", required=True, help="output .tlw8a8 sidecar")
    parser.add_argument("--act-amax", help="optional JSON/NPZ mapping tensor name to activation amax[K]")
    parser.add_argument("--smooth", choices=("auto", "none", "smoothquant"), default="auto")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--smooth-min", type=float, default=1.0e-4)
    parser.add_argument("--smooth-max", type=float, default=1.0e4)
    parser.add_argument("--include-regex", help="override default tensor suffix selection")
    parser.add_argument("--exclude-regex")
    parser.add_argument("--include-output", action="store_true", help="also pack output.weight")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--list", action="store_true", help="list selected tensors without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be in [0, 1]")
    build_sidecar(args)


if __name__ == "__main__":
    main()
