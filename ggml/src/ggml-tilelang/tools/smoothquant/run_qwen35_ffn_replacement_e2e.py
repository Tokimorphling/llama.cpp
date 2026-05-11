#!/usr/bin/env python3
"""Build a Qwen3.5 FFN W8A8 replacement GGUF and benchmark it against native Q8_0."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_REPLACE_RE = r"^blk\.[0-9]+\.ffn_(gate|up|down)\.weight$"
TILELANG_ENV_KEYS = (
    "GGML_TILELANG_W8A8_ENABLE",
    "GGML_TILELANG_W8A8_PRELOAD",
    "GGML_TILELANG_W8A8_GGUF",
    "GGML_TILELANG_W8A8_DEBUG",
    "GGML_TILELANG_SYNC",
    "GGML_TILELANG_QWEN35_FFN_GRAPH",
    "GGML_TILELANG_QWEN35_FFN_GRAPH_DECODE",
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in TILELANG_ENV_KEYS:
        env.pop(key, None)
    return env


def run(
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path = REPO_ROOT,
        log: Path | None = None,
        dry_run: bool = False) -> None:
    print("$ " + shlex.join(str(x) for x in cmd), flush=True)
    if dry_run:
        return

    if log is None:
        subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)
        return

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, [str(x) for x in cmd])


def default_sidecar_path(base_gguf: Path, alpha: float) -> Path:
    alpha_tag = f"{alpha:g}".replace(".", "p")
    return base_gguf.with_name(f"{base_gguf.stem}-ffn-a{alpha_tag}-w8a8.tlw8a8")


def default_replacement_path(base_gguf: Path) -> Path:
    return base_gguf.with_name(f"{base_gguf.stem}-w8a8-ffn-replacement.gguf")


def build_if_requested(args: argparse.Namespace, need_logits: bool) -> None:
    if args.skip_build:
        return

    build_dir = args.build_dir
    if args.configure or not (build_dir / "CMakeCache.txt").exists():
        run([
            "cmake",
            "-S", str(REPO_ROOT),
            "-B", str(build_dir),
            "-DGGML_CUDA=ON",
            "-DGGML_CUDA_TILELANG_INJECTION=ON",
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
        ], dry_run=args.dry_run)

    targets = ["llama-bench"]
    if need_logits:
        targets.append("llama-debug")
    run([
        "cmake",
        "--build", str(build_dir),
        "--target", *targets,
        f"-j{args.jobs}",
    ], dry_run=args.dry_run)


def ensure_sidecar(args: argparse.Namespace, sidecar: Path) -> None:
    if sidecar.exists() and not args.force_sidecar:
        print(f"using existing sidecar: {rel(sidecar)}", flush=True)
        return

    if args.hf_dir is None:
        raise SystemExit(
            f"sidecar does not exist: {sidecar}\n"
            "pass --hf-dir to generate it, or pass --sidecar to an existing .tlw8a8 file"
        )

    cmd = [
        sys.executable,
        str(TOOL_DIR / "convert_qwen_w8a8.py"),
        "--hf-dir", str(args.hf_dir),
        "--gguf", str(args.base_gguf),
        "--out", str(sidecar),
        "--alpha", str(args.alpha),
        "--smooth-min", str(args.smooth_min),
        "--smooth-max", str(args.smooth_max),
        "--max-samples", str(args.max_samples),
        "--seq-len", str(args.seq_len),
        "--device", args.device,
        "--dtype", args.dtype,
        "--include-regex", args.replace_regex,
        "--check-samples", str(args.check_samples),
    ]
    if args.calib is not None:
        cmd += ["--calib", str(args.calib)]
    if args.act_amax is not None:
        cmd += ["--act-amax", str(args.act_amax)]
    if args.smooth_group:
        cmd += ["--smooth-group", args.smooth_group]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    run(cmd, dry_run=args.dry_run)


def ensure_replacement(args: argparse.Namespace, sidecar: Path, replacement: Path) -> None:
    if replacement.exists() and not args.force_replacement:
        print(f"using existing replacement GGUF: {rel(replacement)}", flush=True)
        return

    cmd = [
        sys.executable,
        str(TOOL_DIR / "embed_w8a8_sidecar.py"),
        "--base", str(args.base_gguf),
        "--sidecar", str(sidecar),
        "--output", str(replacement),
        "--name", args.name,
        "--mode", "replacement",
        "--replace-regex", args.replace_regex,
        "--force",
    ]
    run(cmd, dry_run=args.dry_run)


def bench_cmd(binary: Path, model: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        str(binary),
        "-m", str(model),
        "-ngl", str(args.ngl),
        "-p", str(args.prompt_tokens),
        "-n", str(args.gen_tokens),
        "-r", str(args.repetitions),
    ]
    if args.ubatch > 0:
        cmd += ["-ub", str(args.ubatch)]
    if args.no_warmup:
        cmd.append("--no-warmup")
    return cmd + args.bench_extra


def run_benchmarks(args: argparse.Namespace, replacement: Path) -> None:
    bench = args.build_dir / "bin" / "llama-bench"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = clean_env()
    run(
        bench_cmd(bench, args.base_gguf, args),
        env=env,
        log=args.out_dir / "bench-native-q8.log",
        dry_run=args.dry_run,
    )

    env = clean_env()
    if args.tilelang_debug:
        env["GGML_TILELANG_W8A8_DEBUG"] = "1"
    if args.sync:
        env["GGML_TILELANG_SYNC"] = "1"
    run(
        bench_cmd(bench, replacement, args),
        env=env,
        log=args.out_dir / "bench-w8a8-replacement.log",
        dry_run=args.dry_run,
    )


def run_logit_compare(args: argparse.Namespace, replacement: Path) -> None:
    if args.skip_logits:
        return

    debug = args.build_dir / "bin" / "llama-debug"
    base_logits = args.out_dir / "native-q8.logits"
    candidate_logits = args.out_dir / "w8a8-replacement.logits"
    prompt_file = args.out_dir / "logit-prompt.txt"
    prompt_file.write_text(args.logit_prompt, encoding="utf-8")

    base_cmd = [
        str(debug),
        "-m", str(args.base_gguf),
        "-ngl", str(args.ngl),
        "-f", str(prompt_file),
        "--save-logits", str(base_logits),
    ]
    candidate_cmd = [
        str(debug),
        "-m", str(replacement),
        "-ngl", str(args.ngl),
        "-f", str(prompt_file),
        "--save-logits", str(candidate_logits),
    ]
    run(base_cmd, env=clean_env(), log=args.out_dir / "logits-native-q8.log", dry_run=args.dry_run)
    run(candidate_cmd, env=clean_env(), log=args.out_dir / "logits-w8a8-replacement.log", dry_run=args.dry_run)
    run([
        sys.executable,
        str(TOOL_DIR / "compare_logits.py"),
        "--base", str(base_logits),
        "--candidate", str(candidate_logits),
        "--top-k", args.top_k,
    ], log=args.out_dir / "compare-logits.log", dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gguf", required=True, type=Path, help="native Q8_0 GGUF baseline")
    parser.add_argument("--hf-dir", type=Path, help="HF safetensors directory; required only when generating a sidecar")
    parser.add_argument("--sidecar", type=Path, help="input/output .tlw8a8 sidecar path")
    parser.add_argument("--replacement-gguf", type=Path, help="output replacement GGUF path")
    parser.add_argument("--name", default="Qwen3.5-W8A8-SmoothQuant-FFN-Replacement")
    parser.add_argument("--replace-regex", default=DEFAULT_REPLACE_RE)
    parser.add_argument("--force-sidecar", action="store_true", help="regenerate sidecar even if it exists")
    parser.add_argument("--force-replacement", action="store_true", help="regenerate replacement GGUF even if it exists")

    parser.add_argument("--act-amax", type=Path, help="existing SmoothQuant act_amax .npz")
    parser.add_argument("--calib", type=Path, help="plain text, JSONL, or JSON calibration prompts")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--smooth-min", type=float, default=1.0e-4)
    parser.add_argument("--smooth-max", type=float, default=1.0e4)
    parser.add_argument("--smooth-group", default="", help="optional shared groups for sidecar generation, e.g. gate_up")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--check-samples", type=int, default=1)

    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-inject-release")
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--jobs", default=str(os.cpu_count() or 8))
    parser.add_argument("--configure", action="store_true", help="rerun CMake configure before build")
    parser.add_argument("--skip-build", action="store_true")

    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen35-w8a8-ffn-e2e"))
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--ubatch", type=int, default=0, help="pass -ub to llama-bench when > 0")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--bench-extra", nargs="*", default=[], help="extra args appended to llama-bench")
    parser.add_argument("--tilelang-debug", action="store_true")
    parser.add_argument("--sync", action="store_true", help="set GGML_TILELANG_SYNC=1 for the W8A8 benchmark")

    parser.add_argument("--skip-logits", action="store_true")
    parser.add_argument("--logit-prompt", default="Hello")
    parser.add_argument("--top-k", default="1,5,10,50")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_gguf = args.base_gguf.resolve()
    args.build_dir = args.build_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    sidecar = (args.sidecar or default_sidecar_path(args.base_gguf, args.alpha)).resolve()
    replacement = (args.replacement_gguf or default_replacement_path(args.base_gguf)).resolve()

    if not args.base_gguf.is_file() and not args.dry_run:
        raise SystemExit(f"base GGUF does not exist: {args.base_gguf}")
    if args.hf_dir is not None:
        args.hf_dir = args.hf_dir.resolve()

    print(f"repo: {REPO_ROOT}")
    print(f"base_gguf: {rel(args.base_gguf)}")
    print(f"sidecar: {rel(sidecar)}")
    print(f"replacement_gguf: {rel(replacement)}")
    print(f"out_dir: {args.out_dir}")

    build_if_requested(args, need_logits=not args.skip_logits)
    ensure_sidecar(args, sidecar)
    ensure_replacement(args, sidecar, replacement)
    run_logit_compare(args, replacement)
    run_benchmarks(args, replacement)

    print("\nDone.")
    print(f"replacement_gguf: {replacement}")
    print(f"logs: {args.out_dir}")
    print(f"native benchmark: {args.out_dir / 'bench-native-q8.log'}")
    print(f"W8A8 benchmark: {args.out_dir / 'bench-w8a8-replacement.log'}")
    if not args.skip_logits:
        print(f"logit compare: {args.out_dir / 'compare-logits.log'}")


if __name__ == "__main__":
    main()
