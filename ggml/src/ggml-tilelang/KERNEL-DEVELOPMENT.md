# ggml-tilelang Kernel Development

This directory is an experimental TileLang integration area. Keep the runtime
surface small: generated kernels live behind a stable C ABI, model-specific
graph changes stay narrow, and conversion/evaluation scripts stay under
`tools/`.

## Layout

```text
ggml/src/ggml-tilelang/
  tilelang-cuda-injection.cu      ggml-cuda injection runtime for W8A8
  tilelang-cuda-injection.h       injection boundary used by ggml-cuda
  tilelang-kernels.h              C ABI shared by injection and generated kernels
  kernels/
    gen_kernels.py                thin generator CLI
    common.py                     lowering/provenance helpers
    registry.py                   kernel-family registry
    ops/                          TileLang programs and launcher emitters
  generated/                      ignored generated CUDA sources
  tools/smoothquant/              W8A8 conversion, embedding, and eval tools
```

Generated CUDA files are intentionally ignored by git. Edit
`kernels/ops/*.py`, register the family in `kernels/registry.py`, then
regenerate through CMake or `gen_kernels.py`.

## Kernel Workflow

1. Define the operation contract first: dtype, shape, layout, stream behavior,
   and fallback cases.
2. Add a stable C ABI to `tilelang-kernels.h`. Keep CUDA stream arguments as
   `void * stream`.
3. Add the TileLang implementation under `kernels/ops/` and register it in
   `kernels/registry.py`.
4. Wire generated sources in CMake only for the runtime path that actually uses
   them.
5. Verify with the end-to-end `llama-debug`/`llama-bench` wrapper.

Manual generation while iterating:

```bash
python3 ggml/src/ggml-tilelang/kernels/gen_kernels.py \
  --target cuda \
  --kernels w8a8_gemv,w8a8_gemm \
  --out ggml/src/ggml-tilelang/generated
```

## Runtime Scope

The clean Qwen3.5 path is FFN-only W8A8 replacement:

- replacement GGUF stores matched FFN weights as `GGML_TYPE_I8`
- W8A8 scales/smooth tensors are stored as `__tlw8a8.*` auxiliary tensors
- metadata auto-enables TileLang W8A8 and the Qwen3.5 FFN custom graph
- `tilelang-cuda-injection.cu` reuses replacement weight GPU pointers directly
- attention and gated-delta-net layers stay on the native llama.cpp graph

The only model graph environment variables kept for this path are:

```text
GGML_TILELANG_QWEN35_FFN_GRAPH=1
GGML_TILELANG_QWEN35_FFN_GRAPH_DECODE=1
```

Replacement models written by `embed_w8a8_sidecar.py` set these automatically
through GGUF metadata. Normal runs should not need manual environment variables.

Useful debug variables:

```text
GGML_TILELANG_W8A8_DEBUG=1
GGML_TILELANG_SYNC=1
```

## Build

The CUDA injection build keeps the public backend as ggml-cuda and calls
TileLang kernels internally:

```bash
cmake -S . -B build-inject-release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_TILELANG_INJECTION=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-inject-release -j"$(nproc)"
```

## End-to-End Qwen3.5 FFN Replacement

Start with a native Q8_0 GGUF and a calibrated W8A8 sidecar for the same model.
The FFN replacement path expects tensors named like:

```text
blk.<layer>.ffn_gate.weight
blk.<layer>.ffn_up.weight
blk.<layer>.ffn_down.weight
```

The wrapper script can run the full local flow: optional build, optional sidecar
generation, replacement GGUF embedding, optional logit diff, and native Q8_0 vs
W8A8 replacement benchmarks.

Using an existing sidecar:

```bash
python3 ggml/src/ggml-tilelang/tools/smoothquant/run_qwen35_ffn_replacement_e2e.py \
  --base-gguf models-mnt/Qwen3.5-4B/qwen35-4b-q8_0.gguf \
  --sidecar models-mnt/Qwen3.5-4B/qwen35-4b-ffn-a06-w8a8.tlw8a8 \
  --replacement-gguf models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  --ngl 99 \
  --prompt-tokens 512 \
  --gen-tokens 16 \
  --repetitions 3
```

Generating the sidecar from an HF directory first:

```bash
python3 ggml/src/ggml-tilelang/tools/smoothquant/run_qwen35_ffn_replacement_e2e.py \
  --base-gguf models-mnt/Qwen3.5-4B/qwen35-4b-q8_0.gguf \
  --hf-dir models-mnt/Qwen3.5-4B \
  --act-amax models-mnt/Qwen3.5-4B/qwen35-4b-sq-calib128-act-amax.npz \
  --sidecar models-mnt/Qwen3.5-4B/qwen35-4b-ffn-a06-w8a8.tlw8a8 \
  --replacement-gguf models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  --alpha 0.6 \
  --ngl 99
```

Logs are written under `/tmp/qwen35-w8a8-ffn-e2e` by default:

```text
bench-native-q8.log
bench-w8a8-replacement.log
compare-logits.log
```

Pass `--skip-logits` when only benchmark numbers are needed. Pass
`--tilelang-debug` to include the TileLang summary and replacement pointer reuse
counters in the W8A8 benchmark log.

Create a replacement GGUF:

```bash
python3 ggml/src/ggml-tilelang/tools/smoothquant/embed_w8a8_sidecar.py \
  --base models-mnt/Qwen3.5-4B/qwen35-4b-q8_0.gguf \
  --sidecar models-mnt/Qwen3.5-4B/qwen35-4b-ffn-a06-w8a8.tlw8a8 \
  --output models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  --name Qwen3.5-4B-W8A8-SmoothQuant-FFN-Replacement \
  --mode replacement \
  --replace-regex '^blk\.[0-9]+\.ffn_(gate|up|down)\.weight$' \
  --force
```

Run inference:

```bash
./build-inject-release/bin/llama-cli \
  -m models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  -ngl 99 \
  -p "请用一句话介绍杭州。" \
  -n 128
```

The GGUF metadata should auto-enable TileLang. A verbose run should show a small
`uploaded_bytes` value and a large `reused_w_q_bytes` value, meaning replacement
I8 weights were reused from the model buffer instead of copied into a second
TileLang allocation.

## Correctness

Compare logits against the native Q8_0 model:

```bash
./build-inject-release/bin/llama-debug \
  -m models-mnt/Qwen3.5-4B/qwen35-4b-q8_0.gguf \
  -ngl 99 \
  -p "Hello" \
  --save-logits /tmp/q8.logits

./build-inject-release/bin/llama-debug \
  -m models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  -ngl 99 \
  -p "Hello" \
  --save-logits /tmp/w8a8.logits

python3 ggml/src/ggml-tilelang/tools/smoothquant/compare_logits.py \
  /tmp/q8.logits /tmp/w8a8.logits \
  --top-k 1,5,10,50
```

For replacement-vs-carrier checks built from the same sidecar, logits should be
bit-identical. Against native Q8_0, expect small quantization drift.

## Benchmark

Native Q8_0:

```bash
env -u GGML_TILELANG_W8A8_ENABLE \
    -u GGML_TILELANG_W8A8_PRELOAD \
    -u GGML_TILELANG_W8A8_GGUF \
  ./build-inject-release/bin/llama-bench \
    -m models-mnt/Qwen3.5-4B/qwen35-4b-q8_0.gguf \
    -ngl 99 \
    -p 512 \
    -n 16 \
    -r 3
```

W8A8 replacement:

```bash
./build-inject-release/bin/llama-bench \
  -m models-mnt/Qwen3.5-4B/qwen35-4b-w8a8-ffn-replacement.gguf \
  -ngl 99 \
  -p 512 \
  -n 16 \
  -r 3
```

Use `-p` to control prompt/prefill tokens and `-n` to control generated/decode
tokens.

## Porting to Qwen3.5-27B

This code is shape-driven, so Qwen3.5-27B should use the same FFN replacement
path if all of these are true:

- the 27B GGUF uses the same Qwen3.5 architecture and FFN tensor names
- every replacement sidecar tensor is generated from the matching 27B weights
- FFN `K` and `N` satisfy the current W8A8 GEMM constraints (`K % 64 == 0` and
  `N % 64 == 0` for batched prefill)
- the target machine builds TileLang kernels for its CUDA architecture
- enough GPU memory is available for the model, KV cache, compute buffers, and
  W8A8 scale/smooth tensors

Do not reuse a 4B sidecar or replacement GGUF for 27B. Generate a 27B Q8_0 GGUF,
calibrate/pack a 27B W8A8 sidecar, then embed that sidecar in replacement mode
with the same FFN regex.

Single-GPU `-ngl 99` for 27B generally needs substantially more than 24 GB of
VRAM. Multi-GPU placement should be treated as unvalidated for this injection
path until pointer reuse and custom FFN dispatch are checked on each device.

## Review Rules

- Do not edit generated CUDA sources directly.
- Do not widen `supports_op` until unsupported layouts have a tested fallback.
- Keep ad hoc benchmarks out of the core runtime.
- Keep SmoothQuant conversion tools under `tools/smoothquant/`.
- Keep `ggml-cuda.cu` changes thin; TileLang state belongs in
  `tilelang-cuda-injection.cu`.
