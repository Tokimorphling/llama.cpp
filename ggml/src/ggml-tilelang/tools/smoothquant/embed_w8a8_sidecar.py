#!/usr/bin/env python3
"""Build a TileLang W8A8 replacement GGUF from a TLW8A8 sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402


SIDECAR_MAGIC = b"TLW8A8\x01\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a GGUF model, replace selected weights with W8A8 I8 data, and embed TileLang aux tensors.",
    )
    parser.add_argument("--base", required=True, type=Path, help="Input GGUF model")
    parser.add_argument("--sidecar", required=True, type=Path, help="Input .tlw8a8 sidecar")
    parser.add_argument("--output", required=True, type=Path, help="Output fused GGUF path")
    parser.add_argument("--name", default="Qwen3.5-4B-W8A8-SmoothQuant", help="general.name for output GGUF")
    parser.add_argument(
        "--mode",
        choices=("replacement",),
        default="replacement",
        help="only replacement mode is supported",
    )
    parser.add_argument(
        "--replace-regex",
        required=True,
        help="tensor-name regex for weights to replace; unmatched sidecar tensors are dropped",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists")
    return parser.parse_args()


def check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")


def check_sidecar_magic(sidecar: Path) -> None:
    with sidecar.open("rb") as f:
        magic = f.read(len(SIDECAR_MAGIC))
    if magic != SIDECAR_MAGIC:
        raise SystemExit(f"invalid TLW8A8 sidecar magic: {sidecar}")


def read_sidecar_manifest(sidecar: Path) -> tuple[dict, int]:
    check_sidecar_magic(sidecar)
    with sidecar.open("rb") as f:
        f.seek(len(SIDECAR_MAGIC))
        json_len = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(json_len).decode("utf-8"))
    return manifest, int(manifest["data_offset"])


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, name: str) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in {
            gguf.Keys.General.NAME,
            "tilelang.w8a8.version",
            "tilelang.w8a8.storage",
            "tilelang.w8a8.auto_enable",
            "tilelang.w8a8.manifest",
            "tilelang.qwen35.ffn_graph.auto_enable",
            "tilelang.qwen35.ffn_graph.decode",
        }:
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    writer.add_name(name)


def aux_tensor_name(index: int, suffix: str) -> str:
    return f"__tlw8a8.{index}.{suffix}"


def sidecar_array(sidecar: Path, offset: int, nbytes: int, dtype: np.dtype) -> np.memmap:
    return np.memmap(sidecar, mode="r", dtype=dtype, offset=offset, shape=(nbytes // np.dtype(dtype).itemsize,))


def write_gguf_tensors_mode(
        base: Path,
        sidecar: Path,
        output: Path,
        name: str,
        replace_regex: str) -> None:
    manifest, data_offset = read_sidecar_manifest(sidecar)
    reader = gguf.GGUFReader(base)
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise SystemExit(f"base GGUF missing {gguf.Keys.General.ARCHITECTURE}: {base}")

    replace_re = re.compile(replace_regex)

    base_tensor_names = {tensor.name for tensor in reader.tensors}
    manifest_out = json.loads(json.dumps(manifest))
    manifest_out["storage"] = "gguf_replacement_tensors"
    manifest_out["source_model"] = str(base)
    manifest_out["source_format"] = "gguf_w8a8_replacement"

    tensor_entries = []
    sidecar_sources: list[tuple[str, np.ndarray, gguf.GGMLQuantizationType, tuple[int, ...] | None]] = []
    replacement_sources: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType, tuple[int, int]]] = {}
    for item in manifest_out["tensors"]:
        tensor_name = item["name"]
        if replace_re.search(tensor_name) is None:
            continue
        if tensor_name not in base_tensor_names:
            raise SystemExit(f"replacement tensor not found in base GGUF: {tensor_name}")

        index = len(tensor_entries)
        item["index"] = index
        blobs = item["blobs"]
        names = {}
        names["w_q"] = tensor_name
        names["w_scale"] = aux_tensor_name(index, "ws")
        names["smooth_scale"] = aux_tensor_name(index, "sm")
        item["gguf_tensors"] = names

        w_q = blobs["w_q"]
        w_scale = blobs["w_scale"]
        smooth = blobs["smooth_scale"]
        w_q_arr = sidecar_array(sidecar, data_offset + int(w_q["offset"]), int(w_q["nbytes"]), np.int8)
        K = int(item["K"])
        N = int(item["N"])
        replacement_sources[tensor_name] = (
            w_q_arr,
            gguf.GGMLQuantizationType.I8,
            # GGUFWriter reverses tensor shapes when writing tensor-info.
            # Passing (N, K) here stores GGML dimensions {K, N}, while the
            # flat payload remains row-major WQ[N][K] for TileLang.
            (N, K),
        )
        sidecar_sources.append((
            names["w_scale"],
            sidecar_array(sidecar, data_offset + int(w_scale["offset"]), int(w_scale["nbytes"]), np.float32),
            gguf.GGMLQuantizationType.F32,
            None,
        ))
        sidecar_sources.append((
            names["smooth_scale"],
            sidecar_array(sidecar, data_offset + int(smooth["offset"]), int(smooth["nbytes"]), np.float32),
            gguf.GGMLQuantizationType.F32,
            None,
        ))
        tensor_entries.append(item)

    if not replacement_sources:
        raise SystemExit(f"--replace-regex matched no sidecar tensors: {replace_regex}")
    manifest_out["tensors"] = tensor_entries
    writer = gguf.GGUFWriter(output, arch=arch_field.contents(), endianess=reader.endianess)
    copy_metadata(reader, writer, name)
    writer.add_uint32("tilelang.w8a8.version", 1)
    writer.add_string("tilelang.w8a8.storage", manifest_out["storage"])
    writer.add_bool("tilelang.w8a8.auto_enable", True)
    writer.add_bool("tilelang.qwen35.ffn_graph.auto_enable", True)
    writer.add_bool("tilelang.qwen35.ffn_graph.decode", True)
    writer.add_string("tilelang.w8a8.manifest", json.dumps(manifest_out, separators=(",", ":")))

    for tensor in reader.tensors:
        replacement = replacement_sources.get(tensor.name)
        if replacement is not None:
            arr, raw_dtype, shape = replacement
            writer.add_tensor_info(tensor.name, shape, arr.dtype, arr.nbytes, raw_dtype=raw_dtype)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    for tensor_name, arr, raw_dtype, raw_shape in sidecar_sources:
        writer.add_tensor_info(tensor_name, raw_shape or arr.shape, arr.dtype, arr.nbytes, raw_dtype=raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        replacement = replacement_sources.get(tensor.name)
        if replacement is not None:
            arr, _, _ = replacement
            writer.write_tensor_data(arr)
        else:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for _, arr, _, _ in sidecar_sources:
        writer.write_tensor_data(arr)
    writer.close()

    print(f"output={output}")
    print("mode=replacement")
    print(f"base_tensors={len(reader.tensors)}")
    print(f"replacement_tensors={len(replacement_sources)}")
    print(f"w8a8_tensors={len(sidecar_sources)}")
    print(f"sidecar_payload_bytes={sum(arr.nbytes for _, arr, _, _ in sidecar_sources)}")


def main() -> None:
    args = parse_args()
    check_file(args.base, "base GGUF")
    check_file(args.sidecar, "sidecar")
    check_sidecar_magic(args.sidecar)

    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists, pass --force to overwrite: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf_tensors_mode(
        args.base,
        args.sidecar,
        args.output,
        args.name,
        args.replace_regex,
    )


if __name__ == "__main__":
    main()
