#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
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
    ".attn_qkv.weight",
    ".attn_gate.weight",
    ".ssm_out.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
)

HF_QWEN_LAYER_PREFIX_RE = re.compile(r"^(?:model\.language_model|model)\.layers\.(\d+)\.")
GGUF_QWEN_LINEAR_RE = re.compile(
    r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|attn_qkv|attn_gate|ssm_out|ffn_gate|ffn_up|ffn_down)\.weight$")


@dataclass
class SourceTensor:
    name: str
    source_name: str
    source_type: str
    shape: tuple[int, int]  # GGUF order: K, N
    loader: Callable[[], tuple[np.ndarray, str]]


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def sanitize_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name)


def tensor_is_default_linear(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in DEFAULT_SUFFIXES)


def map_hf_qwen_tensor_name(name: str) -> str | None:
    mappings = (
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "blk.{0}.attn_q.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "blk.{0}.attn_k.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "blk.{0}.attn_v.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "blk.{0}.attn_output.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$", "blk.{0}.attn_qkv.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$", "blk.{0}.attn_gate.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.linear_attn\.out_proj\.weight$", "blk.{0}.ssm_out.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "blk.{0}.ffn_gate.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.mlp\.up_proj\.weight$", "blk.{0}.ffn_up.weight"),
        (r"^(?:model\.language_model|model)\.layers\.(\d+)\.mlp\.down_proj\.weight$", "blk.{0}.ffn_down.weight"),
    )
    for pattern, replacement in mappings:
        match = re.match(pattern, name)
        if match:
            return replacement.format(match.group(1))
    if name in ("lm_head.weight", "model.language_model.lm_head.weight"):
        return "output.weight"
    return None


def tensor_to_f32(tensor: Any) -> tuple[np.ndarray, str]:
    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32), qtype.name
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float32), qtype.name

    # BF16 and existing GGUF quantized tensors are exposed as raw bytes by the
    # reader. Dequantize only for explicit prototype packing; production W8A8
    # sidecars should be generated from F16/BF16 source weights.
    return np.asarray(dequantize(tensor.data, qtype), dtype=np.float32), qtype.name


def make_gguf_source_tensor(tensor: Any) -> SourceTensor:
    dims = tuple(int(v) for v in tensor.shape.tolist())
    if len(dims) != 2:
        raise ValueError(f"expected 2D tensor, got {tensor.name} shape={dims}")

    def load() -> tuple[np.ndarray, str]:
        return tensor_to_f32(tensor)

    return SourceTensor(
        name=tensor.name,
        source_name=tensor.name,
        source_type=tensor.tensor_type.name,
        shape=(dims[0], dims[1]),
        loader=load,
    )


def make_safetensors_source_tensor(path: Path, source_name: str, mapped_name: str, shape_nk: list[int]) -> SourceTensor:
    if len(shape_nk) != 2:
        raise ValueError(f"expected 2D tensor, got {source_name} shape={shape_nk}")
    N, K = int(shape_nk[0]), int(shape_nk[1])

    def load() -> tuple[np.ndarray, str]:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors package is required for HF directory input") from exc

        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(source_name)
        source_type = str(tensor.dtype).replace("torch.", "").upper()
        W = tensor.float().cpu().numpy()
        return np.asarray(W, dtype=np.float32), source_type

    return SourceTensor(
        name=mapped_name,
        source_name=source_name,
        source_type="SAFETENSORS",
        shape=(K, N),
        loader=load,
    )


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
    w_amax = np.max(np.abs(W), axis=0).astype(np.float32)
    return compute_smooth_scale_from_amax(w_amax, act_amax, mode, alpha, smooth_min, smooth_max)


def compute_smooth_scale_from_amax(
        w_amax: np.ndarray,
        act_amax: np.ndarray,
        mode: str,
        alpha: float,
        smooth_min: float,
        smooth_max: float) -> tuple[np.ndarray, str]:
    if mode == "none":
        return np.ones(w_amax.size, dtype=np.float32), "none"

    if mode != "smoothquant":
        raise ValueError(f"unknown smooth mode: {mode}")

    w_amax = np.maximum(w_amax, 1.0e-8)
    act_amax = np.maximum(act_amax.astype(np.float32), 1.0e-8)

    smooth = np.power(act_amax, alpha) / np.power(w_amax, 1.0 - alpha)
    smooth = np.clip(smooth, smooth_min, smooth_max).astype(np.float32)
    return smooth, "smoothquant"


def parse_smooth_groups(value: str) -> set[str]:
    if not value:
        return set()
    groups = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(groups - {"qkv", "gate_up"})
    if unknown:
        raise ValueError(f"unsupported smooth groups: {', '.join(unknown)}")
    return groups


def smooth_group_id(name: str, enabled_groups: set[str]) -> str:
    match = GGUF_QWEN_LINEAR_RE.match(name)
    if match is None:
        return ""

    layer, kind = match.group(1), match.group(2)
    if "qkv" in enabled_groups and kind in {"attn_q", "attn_k", "attn_v"}:
        return f"blk.{layer}.attn_qkv"
    if "gate_up" in enabled_groups and kind in {"ffn_gate", "ffn_up"}:
        return f"blk.{layer}.ffn_gate_up"
    return ""


def compute_group_smooth_scales(
        selected: list[SourceTensor],
        act_amax_map: dict[str, np.ndarray],
        enabled_groups: set[str],
        smooth_mode: str,
        alpha: float,
        smooth_min: float,
        smooth_max: float) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if not enabled_groups:
        return {}, {}

    group_w_amax: dict[str, np.ndarray] = {}
    group_act_amax: dict[str, np.ndarray] = {}
    group_info: dict[str, dict[str, Any]] = {}

    for tensor in selected:
        group_id = smooth_group_id(tensor.name, enabled_groups)
        if not group_id:
            continue

        K, N = tensor.shape
        W, _ = tensor.loader()
        if W.shape != (N, K):
            W = W.reshape(N, K)

        w_amax = np.max(np.abs(W), axis=0).astype(np.float32)
        act_amax, act_key = find_act_amax(act_amax_map, tensor.name, K)

        if group_id not in group_w_amax:
            group_w_amax[group_id] = w_amax
            group_act_amax[group_id] = act_amax
            group_info[group_id] = {
                "id": group_id,
                "K": K,
                "members": [],
                "act_amax_sources": [],
            }
        else:
            if group_w_amax[group_id].size != K:
                raise ValueError(f"smooth group {group_id} has mixed K")
            group_w_amax[group_id] = np.maximum(group_w_amax[group_id], w_amax)
            group_act_amax[group_id] = np.maximum(group_act_amax[group_id], act_amax)

        group_info[group_id]["members"].append(tensor.name)
        group_info[group_id]["act_amax_sources"].append(act_key)

    group_scales: dict[str, np.ndarray] = {}
    for group_id, w_amax in group_w_amax.items():
        smooth_scale, applied = compute_smooth_scale_from_amax(
            w_amax, group_act_amax[group_id], smooth_mode, alpha, smooth_min, smooth_max)
        group_scales[group_id] = smooth_scale
        group_info[group_id]["smooth"] = applied
        group_info[group_id]["w_amax_source"] = "max_members"
        group_info[group_id]["act_amax_source"] = "max_members"

    return group_scales, group_info


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


def reference_check(
        name: str,
        W: np.ndarray,
        W_q: np.ndarray,
        w_scale: np.ndarray,
        smooth_scale: np.ndarray,
        n_samples: int) -> dict[str, float | int]:
    if n_samples <= 0:
        return {"n_samples": 0}

    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    max_abs = 0.0
    max_rel = 0.0
    min_cosine = 1.0
    mean_cosine = 0.0

    W_q_f32 = W_q.astype(np.float32)
    for _ in range(n_samples):
        x = rng.standard_normal(W.shape[1], dtype=np.float32)
        y_ref = W @ x

        x_smooth = x / smooth_scale
        x_scale = max(float(np.max(np.abs(x_smooth))) / 127.0, 1.0e-12)
        x_q = np.clip(np.rint(x_smooth / x_scale), -127, 127).astype(np.int8)
        acc = W_q_f32 @ x_q.astype(np.float32)
        y_w8a8 = acc * x_scale * w_scale

        abs_err = np.abs(y_ref - y_w8a8)
        rel_err = abs_err / np.maximum(np.abs(y_ref), 1.0e-6)
        denom = float(np.linalg.norm(y_ref) * np.linalg.norm(y_w8a8))
        cosine = float(np.dot(y_ref, y_w8a8) / denom) if denom > 0.0 else 1.0

        max_abs = max(max_abs, float(np.max(abs_err)) if abs_err.size else 0.0)
        max_rel = max(max_rel, float(np.max(rel_err)) if rel_err.size else 0.0)
        min_cosine = min(min_cosine, cosine)
        mean_cosine += cosine

    return {
        "n_samples": n_samples,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "min_cosine": min_cosine,
        "mean_cosine": mean_cosine / float(n_samples),
    }


def keep_tensor(name: str, source_name: str, args: argparse.Namespace) -> bool:
    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    if include_re is not None:
        if include_re.search(name) is None and include_re.search(source_name) is None:
            return False
    elif not tensor_is_default_linear(name):
        if not (args.include_output and name == "output.weight"):
            return False

    if exclude_re is not None and (exclude_re.search(name) is not None or exclude_re.search(source_name) is not None):
        return False
    return True


def select_gguf_tensors(reader: GGUFReader, args: argparse.Namespace) -> list[SourceTensor]:
    selected = []
    for tensor in reader.tensors:
        if len(tensor.shape) != 2:
            continue
        if not keep_tensor(tensor.name, tensor.name, args):
            continue
        selected.append(make_gguf_source_tensor(tensor))

    if args.max_tensors is not None:
        selected = selected[:args.max_tensors]
    return selected


def select_safetensors_tensors(model_dir: Path, args: argparse.Namespace) -> list[SourceTensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors package is required for HF directory input") from exc

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for path in sorted(model_dir.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = path.name

    selected: list[SourceTensor] = []
    shape_cache: dict[str, Any] = {}
    for source_name, file_name in sorted(weight_map.items()):
        mapped_name = map_hf_qwen_tensor_name(source_name)
        if mapped_name is None:
            continue
        if not keep_tensor(mapped_name, source_name, args):
            continue

        path = model_dir / file_name
        if file_name not in shape_cache:
            with safe_open(path, framework="pt", device="cpu") as f:
                shape_cache[file_name] = {key: f.get_slice(key).get_shape() for key in f.keys()}
        shape = shape_cache[file_name][source_name]
        if len(shape) != 2:
            continue
        selected.append(make_safetensors_source_tensor(path, source_name, mapped_name, shape))

    if args.max_tensors is not None:
        selected = selected[:args.max_tensors]
    return selected


def select_source_tensors(model: Path, args: argparse.Namespace) -> tuple[str, int, int, list[SourceTensor]]:
    source_stat = model.stat()
    if model.is_dir():
        selected = select_safetensors_tensors(model, args)
        source_size = sum(p.stat().st_size for p in model.glob("*.safetensors"))
        source_mtime_ns = max((p.stat().st_mtime_ns for p in model.glob("*.safetensors")), default=source_stat.st_mtime_ns)
        return "hf_safetensors", source_size, source_mtime_ns, selected

    reader = GGUFReader(model)
    selected = select_gguf_tensors(reader, args)
    return "gguf", source_stat.st_size, source_stat.st_mtime_ns, selected


def load_gguf_shapes(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}

    reader = GGUFReader(path)
    shapes: dict[str, tuple[int, int]] = {}
    for tensor in reader.tensors:
        dims = tuple(int(v) for v in tensor.shape.tolist())
        if len(dims) == 2:
            shapes[tensor.name] = (dims[0], dims[1])
    return shapes


def validate_against_gguf(selected: list[SourceTensor], gguf_shapes: dict[str, tuple[int, int]]) -> None:
    if not gguf_shapes:
        return

    missing: list[str] = []
    mismatched: list[str] = []
    for tensor in selected:
        expected = gguf_shapes.get(tensor.name)
        if expected is None:
            missing.append(tensor.name)
            continue
        if expected != tensor.shape:
            mismatched.append(f"{tensor.name}: source={tensor.shape} gguf={expected}")

    if missing or mismatched:
        details = []
        if missing:
            details.append("missing in GGUF: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
        if mismatched:
            details.append("shape mismatch: " + "; ".join(mismatched[:8]) + (" ..." if len(mismatched) > 8 else ""))
        raise ValueError("selected sidecar tensors do not match GGUF: " + " | ".join(details))


def append_blob(spool: Any, blob_offset: int, array: np.ndarray) -> tuple[dict[str, Any], int]:
    data = np.ascontiguousarray(array).tobytes(order="C")
    spool.write(data)
    return {
        "offset": blob_offset,
        "nbytes": len(data),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }, blob_offset + len(data)


def write_sidecar(path: Path, manifest: dict[str, Any], blob_path: Path) -> None:
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
        with blob_path.open("rb") as blob_file:
            while True:
                chunk = blob_file.read(64 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def build_sidecar(args: argparse.Namespace) -> None:
    model = Path(args.model)
    output = Path(args.output)
    source_format, source_size, source_mtime_ns, selected = select_source_tensors(model, args)
    gguf_path = Path(args.gguf) if args.gguf else None
    gguf_shapes = load_gguf_shapes(gguf_path)
    validate_against_gguf(selected, gguf_shapes)
    if args.list:
        for tensor in selected:
            print(f"{tensor.name}\t{tensor.source_type}\tK={tensor.shape[0]}\tN={tensor.shape[1]}\t{tensor.source_name}")
        return

    if not selected:
        raise SystemExit("no tensors selected")

    act_amax_map = load_act_amax(Path(args.act_amax) if args.act_amax else None)
    smooth_mode = args.smooth
    if smooth_mode == "auto":
        smooth_mode = "smoothquant" if act_amax_map else "none"
    enabled_smooth_groups = parse_smooth_groups(args.smooth_group)
    group_smooth_scales, group_info = compute_group_smooth_scales(
        selected,
        act_amax_map,
        enabled_smooth_groups,
        smooth_mode,
        args.alpha,
        args.smooth_min,
        args.smooth_max,
    )

    manifest: dict[str, Any] = {
        "format": "ggml-tilelang-w8a8-sidecar",
        "version": 1,
        "data_alignment": DATA_ALIGNMENT,
        "source_format": source_format,
        "source_model": str(model),
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "gguf_model": str(gguf_path) if gguf_path else "",
        "gguf_validated": bool(gguf_shapes),
        "created_unix": int(time.time()),
        "quantization": {
            "recipe": "smoothquant_rtn_w8a8",
            "smooth": smooth_mode,
            "alpha": args.alpha,
            "activation": "dynamic_per_token_int8",
            "activation_scale": "per_token_float32",
            "weight": "int8",
            "weight_scale": "per_output_channel_float32",
            "smooth_group": ",".join(sorted(enabled_smooth_groups)),
        },
        "layout": {
            "weight_q": "row_major_int8_N_by_K",
            "weight_scale": "float32_N",
            "smooth_scale": "float32_K",
        },
        "smooth_groups": [],
        "tensors": [],
    }
    blob_tmp = output.with_suffix(output.suffix + ".blobs.tmp")

    total_wq_bytes = 0
    total_blob_bytes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with blob_tmp.open("wb") as spool:
            group_smooth_blobs: dict[str, dict[str, Any]] = {}
            for idx, tensor in enumerate(selected):
                K, N = tensor.shape
                W, loaded_type = tensor.loader()
                if W.shape != (N, K):
                    W = W.reshape(N, K)

                act_amax, act_key = find_act_amax(act_amax_map, tensor.name, K)
                group_id = smooth_group_id(tensor.name, enabled_smooth_groups)
                if group_id and group_id in group_smooth_scales:
                    smooth_scale = group_smooth_scales[group_id]
                    applied_smooth = group_info[group_id]["smooth"]
                    act_key = "group:" + group_id
                else:
                    group_id = ""
                    smooth_scale, applied_smooth = compute_smooth_scale(
                        W, act_amax, smooth_mode, args.alpha, args.smooth_min, args.smooth_max)
                W_q, w_scale, stats = quantize_weight(W, smooth_scale)
                check = reference_check(tensor.name, W, W_q, w_scale, smooth_scale, args.check_samples)

                w_q_blob, total_blob_bytes = append_blob(spool, total_blob_bytes, W_q)
                w_scale_blob, total_blob_bytes = append_blob(spool, total_blob_bytes, w_scale)
                if group_id:
                    if group_id not in group_smooth_blobs:
                        smooth_blob, total_blob_bytes = append_blob(spool, total_blob_bytes, smooth_scale.astype(np.float32))
                        group_smooth_blobs[group_id] = smooth_blob
                    else:
                        smooth_blob = group_smooth_blobs[group_id]
                else:
                    smooth_blob, total_blob_bytes = append_blob(spool, total_blob_bytes, smooth_scale.astype(np.float32))
                total_wq_bytes += w_q_blob["nbytes"]

                tensor_entry = {
                    "index": idx,
                    "name": tensor.name,
                    "source_name": tensor.source_name,
                    "source_type": loaded_type,
                    "source_shape": [K, N],
                    "K": K,
                    "N": N,
                    "source_sha256": stable_sha256(W),
                    "act_amax_source": act_key,
                    "smooth": applied_smooth,
                    "smooth_group_id": group_id,
                    "blobs": {
                        "w_q": w_q_blob,
                        "w_scale": w_scale_blob,
                        "smooth_scale": smooth_blob,
                    },
                    "stats": stats,
                    "reference_check": check,
                }
                manifest["tensors"].append(tensor_entry)

                print(
                    f"packed {tensor.name}: source={tensor.source_name} type={loaded_type} K={K} N={N} "
                    f"smooth={applied_smooth} group={group_id or '-'} qerr_max={stats['weight_quant_max_abs_err']:.6g} "
                    f"cos={check.get('min_cosine', 0.0):.8f}",
                    file=sys.stderr,
                )

            manifest["smooth_groups"] = [
                {
                    **group_info[group_id],
                    "blobs": {
                        "smooth_scale": group_smooth_blobs[group_id],
                    },
                }
                for group_id in sorted(group_smooth_blobs)
            ]

        manifest["summary"] = {
            "n_tensors": len(manifest["tensors"]),
            "n_smooth_groups": len(manifest["smooth_groups"]),
            "w_q_bytes": total_wq_bytes,
            "total_blob_bytes": total_blob_bytes,
        }
        write_sidecar(output, manifest, blob_tmp)
        print(json.dumps({
            "output": str(output),
            "n_tensors": len(manifest["tensors"]),
            "total_blob_bytes": manifest["summary"]["total_blob_bytes"],
        }, indent=2))
    finally:
        if blob_tmp.exists():
            blob_tmp.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack GGUF or HF safetensors linear weights into a TileLang SmoothQuant-style W8A8 sidecar.")
    parser.add_argument("--model", required=True, help="source F16/BF16 GGUF or HF safetensors model directory")
    parser.add_argument("--output", required=True, help="output .tlw8a8 sidecar")
    parser.add_argument("--gguf", help="optional GGUF used to validate runtime tensor names and K/N shapes")
    parser.add_argument("--act-amax", help="optional JSON/NPZ mapping tensor name to activation amax[K]")
    parser.add_argument("--smooth", choices=("auto", "none", "smoothquant"), default="auto")
    parser.add_argument("--smooth-group", default="",
                        help="comma-separated shared SmoothQuant groups to pack: qkv,gate_up")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--smooth-min", type=float, default=1.0e-4)
    parser.add_argument("--smooth-max", type=float, default=1.0e4)
    parser.add_argument("--include-regex", help="override default tensor suffix selection")
    parser.add_argument("--exclude-regex")
    parser.add_argument("--include-output", action="store_true", help="also pack output.weight")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--check-samples", type=int, default=1, help="number of deterministic random x vectors for FP32 reference checks")
    parser.add_argument("--list", action="store_true", help="list selected tensors without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be in [0, 1]")
    build_sidecar(args)


if __name__ == "__main__":
    main()
