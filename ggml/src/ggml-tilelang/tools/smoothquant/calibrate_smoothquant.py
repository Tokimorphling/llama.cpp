#!/usr/bin/env python3
"""Collect SmoothQuant activation amax vectors for Qwen-style Linear weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pack_w8a8_sidecar import map_hf_qwen_tensor_name, sanitize_key


DEFAULT_PROMPTS = [
    "Hello",
    "\u8bf7\u7528\u4e00\u53e5\u8bdd\u4ecb\u7ecd\u676d\u5dde\u3002",
    "def fibonacci(n):\n    ",
    "Question: If there are 12 apples and you give away 5, how many are left?\nAnswer:",
    (
        "In a small engineering notebook, the team recorded the same observation three times: "
        "first keep the numerical result stable, then measure speed, then change packaging. "
        "Summarize the rule in one sentence."
    ),
]

TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def load_prompts(path: Path | None, max_samples: int) -> list[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON prompt file must be a list")
        prompts = [str(item.get("prompt", item.get("text", item))) if isinstance(item, dict) else str(item) for item in data]
    else:
        prompts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    obj = json.loads(line)
                    prompts.append(str(obj.get("prompt", obj.get("text", ""))))
                else:
                    prompts.append(line)

    prompts = [p for p in prompts if p]
    if max_samples > 0:
        prompts = prompts[:max_samples]
    if not prompts:
        raise ValueError("no calibration prompts")
    return prompts


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_transformers_model(model_dir: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for calibration") from exc

    dtype = torch_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if dtype != "auto":
        kwargs["dtype"] = dtype
    else:
        kwargs["dtype"] = torch.float16 if args.device.startswith("cuda") else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "model type `qwen3_5`" in message or "model type 'qwen3_5'" in message:
            raise RuntimeError(
                "transformers in this environment does not recognize model_type=qwen3_5. "
                "Install a transformers build that supports Qwen3.5, or provide local "
                "trust_remote_code model files, then rerun this calibration script."
            ) from exc
        raise

    model.eval()
    if args.device != "auto":
        model.to(args.device)
    elif torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def target_weight_name(module_name: str) -> str | None:
    if not any(module_name.endswith(suffix) for suffix in TARGET_SUFFIXES):
        return None
    return map_hf_qwen_tensor_name(module_name + ".weight")


def collect_targets(model: Any) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        mapped = target_weight_name(module_name)
        if mapped is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        targets[mapped] = module
    return targets


def update_amax(current: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if not torch.is_floating_point(x):
        x = x.float()
    x = x.abs().reshape(-1, x.shape[-1]).amax(dim=0).float().cpu()
    if current is None:
        return x
    return torch.maximum(current, x)


def run_calibration(args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.prompts) if args.prompts else None, args.max_samples)
    tokenizer, model = load_transformers_model(Path(args.model), args)
    targets = collect_targets(model)
    if args.list_targets:
        for name in sorted(targets):
            module = targets[name]
            print(f"{name}\tin={module.in_features}\tout={module.out_features}")
        return
    if not targets:
        raise RuntimeError("no Qwen-style target Linear modules found")

    act_amax: dict[str, torch.Tensor | None] = {name: None for name in targets}
    hooks = []
    for name, module in targets.items():
        def make_hook(key: str):
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                if not inputs:
                    return
                act_amax[key] = update_amax(act_amax[key], inputs[0])
            return hook
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    device = next(model.parameters()).device
    try:
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            print(f"calibrated {idx + 1}/{len(prompts)} tokens={encoded['input_ids'].shape[-1]}", file=sys.stderr)
    finally:
        for hook in hooks:
            hook.remove()

    arrays: dict[str, np.ndarray] = {}
    for name, value in act_amax.items():
        if value is None:
            continue
        array = value.numpy().astype(np.float32)
        arrays[name] = array
        arrays[sanitize_key(name)] = array

    if not arrays:
        raise RuntimeError("calibration produced no activation amax values")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(output.suffix + ".manifest.json")
    manifest = {
        "format": "ggml-tilelang-smoothquant-act-amax",
        "version": 1,
        "created_unix": int(time.time()),
        "model": str(args.model),
        "prompts": str(args.prompts) if args.prompts else "builtin",
        "num_prompts": len(prompts),
        "seq_len": args.seq_len,
        "target_count": len(targets),
        "output": str(output),
        "keys": sorted(k for k in arrays if not k.startswith("blk_")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "n_arrays": len(arrays)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="output .npz with act_amax vectors")
    parser.add_argument("--manifest", help="optional calibration manifest JSON")
    parser.add_argument("--prompts", help="plain text, JSONL, or JSON calibration prompt file")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--list-targets", action="store_true", help="load model and list calibration targets")
    return parser.parse_args()


def main() -> None:
    run_calibration(parse_args())


if __name__ == "__main__":
    main()
