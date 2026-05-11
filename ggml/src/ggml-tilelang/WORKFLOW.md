# ggml-tilelang Operator Porting Workflow

This document is the working playbook for adding TileLang-backed operators to
`ggml-tilelang`. Keep every new operator small enough to pass through this
pipeline before widening shape support or optimizing performance.

## Scope

`ggml-tilelang` has two kernel paths:

- CUDA reference stub: used for first correctness bring-up.
- TileLang AOT: generated CUDA device source plus a stable C ABI launcher.

The C++ backend must call only the C ABI declared in `tilelang-kernels.h`. It
must not know whether the implementation is the reference stub or TileLang AOT.

`ggml-tilelang` also has two execution modes:

- Composite backend, default: TileLang owns the external scheduler split and
  delegates unsupported nodes to an internal ggml-cuda backend.
- Native-only backend, debug: set `GGML_TILELANG_CUDA_DELEGATE=0` to recover the
  old narrow-backend behavior where only TileLang-native ops are claimed.

Composite mode is the performance path. It keeps tensors in
`ggml_backend_cuda_buffer_type(0)`, runs CUDA delegate subgraphs on the internal
CUDA backend stream, and launches TileLang native kernels on the same stream.

## Operator Porting Checklist

For each new operator or dtype variant:

1. Define the exact narrow semantics.
   - Example: `Q8_0 weight + F32 activation -> F32 output`.
   - Specify tensor shapes, contiguous requirements, dtype requirements, and
     rejected batching/transposition cases.

2. Add one stable C ABI function in `tilelang-kernels.h`.
   - Keep CUDA types out of the header when possible; use `void * stream`.
   - Keep the signature identical between stub and AOT.

3. Add a strict `supports_op` predicate.
   - Start narrower than the eventual target.
   - Require supported dtypes, output dtype, contiguity, shape, and layout.
   - Reject unsupported batch dimensions explicitly.

4. Add backend dispatch and counters.
   - Dispatch by op and weight dtype.
   - Add a per-kernel call counter.
   - Add reject counters for disabled, unsupported dtype, layout, batch, and
     shape guards.
   - Log the selected kernel path once: `CUDA reference stub` or `AOT TileLang`.

5. Implement the CUDA reference stub.
   - Make it simple and readable, not fast.
   - Use repository layout definitions when available, for example
     `GGML_COMMON_DECL_CUDA` plus `ggml-common.h` for quantized block structs.
   - Until the backend passes a real ggml-cuda stream, synchronize around the
     null-stream launch for correctness. Remove that guard only when stream
     ordering is explicit.

6. Extend `kernels/gen_kernels.py`.
   - Add the TileLang program.
   - Add `--kernels` support for selective generation if needed.
   - Emit provenance headers in all generated files:
     generator path, TileLang version, target, command, and do-not-edit marker.

7. Add AOT sources to CMake.
   - When `GGML_TILELANG_USE_AOT_KERNELS=ON`, check generated files exist and
     fail with a command telling the developer to run the generator.
   - When OFF, compile only the reference stubs.

## Correctness Gate

Correctness always runs in both paths.

### 1. Generate AOT files

Use the conda base environment for TileLang:

```bash
conda run -n base python ggml/src/ggml-tilelang/kernels/gen_kernels.py \
  --target cuda \
  --kernels f16_gemv,q8_0_gemv \
  --out ggml/src/ggml-tilelang/generated
```

### 2. Build and test the stub path

```bash
cmake -B build-stub \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=OFF \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build-stub --target test-backend-ops -j8

GGML_TILELANG_ENABLE=1 GGML_TILELANG_NATIVE_POLICY=all \
  ./build-stub/bin/test-backend-ops test -o "MUL_MAT" -b TileLang
```

Expected:

- The log says `ggml-tilelang: kernel path = CUDA reference stub`.
- All supported cases pass.
- The new operator counter is greater than zero.

### 3. Build and test the AOT path

```bash
cmake -B build-aot \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build-aot --target test-backend-ops -j8

GGML_TILELANG_ENABLE=1 GGML_TILELANG_NATIVE_POLICY=all \
  ./build-aot/bin/test-backend-ops test -o "MUL_MAT" -b TileLang
```

Expected:

- The log says `ggml-tilelang: kernel path = AOT TileLang`.
- Stub and AOT pass the same supported cases.
- Stub and AOT counters match for the new operator.

### 4. Verify runtime opt-in

Without `GGML_TILELANG_ENABLE`, TileLang must not claim ops:

```bash
./build-aot/bin/test-backend-ops test -o "MUL_MAT" -b TileLang
./build-stub/bin/test-backend-ops test -o "MUL_MAT" -b TileLang
```

Expected:

- `0/0 tests passed`.
- All TileLang operator counters are zero.

### 5. Useful log extraction

```bash
rg "kernel path|MUL_MAT_.* calls|tests passed" /tmp/ggml-tilelang-*.log
```

For the current F16 + Q8_0 milestone, the expected enabled result is:

```text
MUL_MAT_F16 calls = 7
MUL_MAT_Q8_0 calls = 4
11/11 tests passed
```

## Real GGUF Smoke Gate

Run this gate after `test-backend-ops` passes. It verifies that TileLang can be
scheduled inside a real llama.cpp graph and that unsupported shapes fall back.

### Device ordering

Composite mode can be used as the primary device because it internally creates a
CUDA delegate backend:

```bash
-dev TileLang
```

The old scheduler-overlay mode can still be tested by disabling the delegate:

```bash
GGML_TILELANG_CUDA_DELEGATE=0 \
  ./build-aot/bin/llama-completion ... -dev TileLang,CUDA0 -ts 0,1
```

That mode is useful for isolating native kernels, but it creates many external
CUDA <-> TileLang graph splits and should not be used for headline throughput.

### Composite native policy

Composite mode defaults to a conservative native policy:

```bash
GGML_TILELANG_NATIVE_POLICY=large_n
```

This claims only supported `MUL_MAT` nodes whose weight matrix has
`N >= 8192`. For the tiny F16/Q8_0 models this hits the output projection and
leaves FFN/GLU and other fuseable regions inside the CUDA delegate. This avoids
breaking CUDA fusion while keeping TileLang native calls greater than zero.

Useful policy knobs:

```bash
GGML_TILELANG_NATIVE=0              # delegate-only composite smoke
GGML_TILELANG_NATIVE_POLICY=large_n # default composite performance path
GGML_TILELANG_NATIVE_POLICY=all     # native kernel correctness stress
GGML_TILELANG_NATIVE_POLICY=none    # same as native disabled
```

Use `all` for `test-backend-ops` and microbench correctness. Use the default
`large_n` policy for real GGUF deterministic and performance runs unless the
current experiment explicitly studies fusion disruption.

### Shape and reject diagnostics

Enable limited shape logging when validating real models:

```bash
GGML_TILELANG_ENABLE=1 \
GGML_TILELANG_DUMP_SHAPES=1 \
  ./build-aot/bin/llama-bench -v \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang \
  -ngl 99 \
  -p 32 \
  -n 16
```

Expected diagnostics:

- Hit lines show `MUL_MAT_F16` or `MUL_MAT_Q8_0` with `K`, `N`, `ne`, and `nb`.
- Hit lines must show a non-null `stream=0x...` in real model runs and direct
  `test-backend-ops -b TileLang` runs. Composite mode owns an internal CUDA
  backend, so null stream fallback is not expected.
- Reject lines explain fallback, for example `non-contiguous`, `batch != 1`,
  `K % 32 != 0`, `shape mismatch`, or `disabled`.
- End-of-context counters show nonzero target calls for enabled runs.

### Stream normalization

TileLang reuses `ggml_backend_cuda_buffer_type(device)` for zero-copy access, but
buffer compatibility does not imply stream ordering. The execution backend must
launch TileLang kernels on the same CUDA stream as ggml-cuda for the matching
device.

The composite stream path is:

```text
TileLang backend init
  -> ggml_backend_cuda_init(device)
  -> ggml_backend_cuda_get_stream(internal_cuda_backend)
TileLang graph compute
  -> delegate CUDA-only subgraphs to internal CUDA backend
  -> tilelang_*_gemv(..., stream)
```

The kernel launchers do not synchronize by default. Use this only as a debug
guard:

```bash
GGML_TILELANG_SYNC=1
```

M3.3 regression matrix:

```bash
# AOT, no debug sync
GGML_TILELANG_ENABLE=1 ./build-aot/bin/test-backend-ops test -o "MUL_MAT" -b TileLang

# AOT, debug sync
GGML_TILELANG_ENABLE=1 GGML_TILELANG_SYNC=1 \
  ./build-aot/bin/test-backend-ops test -o "MUL_MAT" -b TileLang

# Stub, no debug sync
GGML_TILELANG_ENABLE=1 ./build-stub/bin/test-backend-ops test -o "MUL_MAT" -b TileLang
```

Then repeat model smoke and deterministic token smoke for F16 and Q8_0. If a
real model run shows `stream=(nil)`, the backend has fallen back to the null
stream and the run is not a valid stream-normalized benchmark.

The stream hook is guarded as experimental:

```bash
cmake -B build-aot \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=ON \
  -DGGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP=ON
```

With interop enabled, hit logs should include matching device ids:

```text
tilelang_device=0 stream_device=0 src0_device=0 src1_device=0 dst_device=0
```

Any mismatch is a correctness failure, not a performance fallback.

Sanitizer notes:

```bash
GGML_TILELANG_ENABLE=1 \
  compute-sanitizer ./build-aot/bin/test-backend-ops test -o "MUL_MAT" -b TileLang

GGML_TILELANG_ENABLE=1 \
  compute-sanitizer --tool racecheck ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 4 --temp 0 --seed 1234 -no-cnv --no-display-prompt
```

`racecheck` can report existing CUDA backend hazards in kernels such as softmax.
Always compare against a CUDA-only baseline before attributing a racecheck report
to TileLang.

### Stub and AOT model smoke

Run the same tiny F16 and Q8_0 models through both kernel paths:

```bash
GGML_TILELANG_ENABLE=1 GGML_TILELANG_DUMP_SHAPES=1 \
  ./build-stub/bin/llama-completion \
  -m models/tilelang/tiny-f16.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 16 --temp 0 --seed 1234 -no-cnv --no-display-prompt

GGML_TILELANG_ENABLE=1 GGML_TILELANG_DUMP_SHAPES=1 \
  ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 16 --temp 0 --seed 1234 -no-cnv --no-display-prompt
```

Acceptance:

- F16 and Q8_0 runs finish without CUDA errors.
- Stub and AOT both show nonzero counters for the target dtype.
- Running without `GGML_TILELANG_ENABLE` finishes with TileLang call counters at
  zero and `disabled` rejects greater than zero.
- `llama-bench` and `llama-completion` use different device-list delimiters in
  this tree (`/` vs `,`). Always trust counters, not just the `dev` column.

### Deterministic token smoke

Compare a CUDA baseline with TileLang stub and AOT. Use `llama-completion`, not
`llama-cli`, to avoid interactive mode:

```bash
./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev CUDA0 \
  -ngl 99 \
  -p "Once upon a time" \
  -n 16 \
  --temp 0 \
  --seed 1234 \
  -no-cnv \
  --no-display-prompt \
  > /tmp/cuda.out 2> /tmp/cuda.err

GGML_TILELANG_ENABLE=1 ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang \
  -ngl 99 \
  -p "Once upon a time" \
  -n 16 \
  --temp 0 \
  --seed 1234 \
  -no-cnv \
  --no-display-prompt \
  > /tmp/tilelang-aot.out 2> /tmp/tilelang-aot.err

cmp /tmp/cuda.out /tmp/tilelang-aot.out
```

Repeat with `build-stub`. The generated text should match byte-for-byte for the
current narrow GEMV path. If sampling asserts or output diverges, first check
CUDA stream ordering and then tensor layout assumptions.

## Support Coverage Probe

Use support mode when changing `supports_op`:

```bash
GGML_TILELANG_ENABLE=1 \
  ./build-aot/bin/test-backend-ops support -o "MUL_MAT" -b TileLang --output csv
```

This is useful for confirming that a new predicate supports only the intended
cases. Do not widen support just to increase the count; widen only when the
kernel handles the shape/layout correctly.

## Benchmark Gate

Benchmark only after correctness is stable.

### Q8_0 Variant And Timing Knobs

The Q8_0 AOT launcher currently has three variants:

```bash
# one thread computes one output row; default baseline
GGML_TILELANG_Q8_0_VARIANT=thread

# one CUDA block computes one output row; alias: row1
GGML_TILELANG_Q8_0_VARIANT=block

# one warp computes one output row
GGML_TILELANG_Q8_0_VARIANT=warp
```

Use CUDA event timing only when measuring kernel time. It records events around
each TileLang Q8_0 launch and collects elapsed time when the backend is freed,
so it is suitable for kernel-time attribution. Keep clean end-to-end throughput
runs separate and do not enable timing for the headline token/s number:

```bash
GGML_TILELANG_TIMING=1
```

The backend prints:

```text
MUL_MAT_Q8_0 kernel time total=... ms calls=... avg=... ms
Q8_0 hit shape histogram:
Q8_0 reject shape histogram:
```

For token-level smoke, collect throughput without `GGML_TILELANG_TIMING`, then
repeat with timing enabled only to get kernel event time:

```bash
./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev CUDA0 -ngl 99 \
  -p "Once upon a time" -n 64 --temp 0 --seed 1234 -no-cnv --no-display-prompt

GGML_TILELANG_ENABLE=1 \
  ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 64 --temp 0 --seed 1234 -no-cnv --no-display-prompt

GGML_TILELANG_ENABLE=1 GGML_TILELANG_Q8_0_VARIANT=block \
  ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 64 --temp 0 --seed 1234 -no-cnv --no-display-prompt

GGML_TILELANG_ENABLE=1 GGML_TILELANG_TIMING=1 \
  ./build-aot/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Once upon a time" -n 64 --temp 0 --seed 1234 -no-cnv --no-display-prompt
```

### Q8_0 standalone GEMV microbench

Use the standalone microbench to separate kernel behavior from llama.cpp graph
splits, scheduler decisions, and tiny-model launch overhead:

```bash
cmake -B build-aot-release \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=ON \
  -DGGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-aot-release --target tilelang-q8-0-microbench -j8

./build-aot-release/bin/tilelang-q8-0-microbench \
  --warmup 3 \
  --iters 10
```

The default shape set covers decode-sized Q8_0 GEMV cases:

```text
K=1024  N=1024
K=2048  N=2048
K=4096  N=4096
K=4096  N=11008
K=4096  N=32000
K=8192  N=8192
K=8192  N=28672
```

The output is CSV:

```text
K,N,variant,time_ms,logical_GBps,max_abs_err,max_rel_err
```

Compare `tilelang_thread`, `tilelang_block`, and `tilelang_warp` against
`cuda_ref`. If the standalone kernel is close to `cuda_ref` but model-level
token/s is poor, inspect graph splits, scheduler copies, and backend boundary
overhead before changing the kernel.

### test-backend-ops microbench

Use `test-backend-ops perf` for operator-level comparisons:

```bash
cmake -B build-aot-release \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-aot-release --target test-backend-ops -j8

GGML_TILELANG_ENABLE=1 \
  ./build-aot-release/bin/test-backend-ops perf -o "MUL_MAT" -b TileLang --output csv
```

Run a CUDA baseline from the same build type and machine state:

```bash
./build-aot-release/bin/test-backend-ops perf -o "MUL_MAT" -b CUDA --output csv
```

Rules:

- Compare Release to Release, not Debug to Release.
- Capture the full command, git commit, GPU, CUDA version, and generated header.
- Repeat after a warm machine state; ignore the first result if it is an outlier.
- Keep correctness logs with benchmark logs.
- If performance regresses, keep the correctness kernel and optimize in a
  separate milestone.

### Model-level benchmark

Use `llama-bench` only after a model path actually hits TileLang counters.
Build Release:

```bash
cmake -B build-aot-release \
  -DGGML_TILELANG=ON \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG_USE_AOT_KERNELS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-aot-release --target llama-bench -j8
```

Run a tiny, repeatable case first:

```bash
GGML_TILELANG_ENABLE=1 \
  ./build-aot-release/bin/llama-bench \
  -m /path/to/model-q8_0.gguf \
  -dev TileLang \
  -p 128 \
  -n 32 \
  -ngl 99 \
  -r 5 \
  -o json
```

Run the same command without `GGML_TILELANG_ENABLE` for the baseline. The
TileLang run is meaningful only if a matching `llama-completion` smoke for the
same model shows nonzero counters for the target operator. Some `llama-bench`
paths suppress backend counter logs, so keep a completion smoke log with every
benchmark result.

### Composite split-reduction check

Use this gate after changing delegate or native policy code:

```bash
./build-aot-release/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev CUDA0 -ngl 99 \
  -p "Hello" -n 64 --temp 0 --seed 123 --no-perf \
  > /tmp/cuda.out 2> /tmp/cuda.err

GGML_TILELANG_ENABLE=1 GGML_TILELANG_NATIVE=0 \
  ./build-aot-release/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Hello" -n 64 --temp 0 --seed 123 --no-perf \
  > /tmp/delegate-only.out 2> /tmp/delegate-only.err

GGML_TILELANG_ENABLE=1 GGML_TILELANG_Q8_0_VARIANT=warp \
  ./build-aot-release/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev TileLang -ngl 99 \
  -p "Hello" -n 64 --temp 0 --seed 123 --no-perf \
  > /tmp/tilelang.out 2> /tmp/tilelang.err

cmp /tmp/cuda.out /tmp/delegate-only.out
cmp /tmp/cuda.out /tmp/tilelang.out
rg "graph splits|MUL_MAT_.* calls|CUDA delegate runs|device_mismatch" /tmp/*.err
```

Expected signs:

```text
CUDA baseline:       graph splits = 2
delegate-only:       graph splits = 1, native calls = 0, delegate nodes > 0
default composite:   graph splits = 1, Q8_0/F16 calls > 0, delegate nodes > 0
device_mismatch = 0
```

If `GGML_TILELANG_NATIVE_POLICY=all` diverges but `large_n` matches, the
delegate path is sound and the divergence is caused by disrupting CUDA fusion
regions. Keep `all` as a correctness stress mode, not a throughput mode.

### CUDA injection split check

Use CUDA injection mode to test whether backend boundaries, not the TileLang
kernel, are dominating model-level performance. This is an experiment build, not
the separate TileLang backend build:

```bash
cmake -B build-cuda-inject-release \
  -DGGML_CUDA=ON \
  -DGGML_TILELANG=OFF \
  -DGGML_CUDA_TILELANG_INJECTION=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda-inject-release --target llama-completion test-backend-ops -j8
```

Run the model through CUDA only; `GGML_TILELANG_ENABLE=1` enables the internal
TileLang override inside ggml-cuda:

```bash
./build-cuda-inject-release/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev CUDA0 -ngl 99 \
  -p "Once upon a time" -n 128 --temp 0 --seed 1234 --perf

GGML_TILELANG_ENABLE=1 GGML_TILELANG_Q8_0_VARIANT=warp \
  ./build-cuda-inject-release/bin/llama-completion \
  -m models/tilelang/tiny-q8_0.gguf \
  -dev CUDA0 -ngl 99 \
  -p "Once upon a time" -n 128 --temp 0 --seed 1234 --perf
```

Expected signs:

```text
sched_reserve: graph splits = 2
ggml-cuda: TileLang injection MUL_MAT_Q8_0 calls = ...
```

If injection has `graph splits` close to CUDA baseline and recovers token/s, the
next architecture task is reducing external scheduler boundaries, for example a
composite backend. If injection remains slow, go back to kernel and memory-level
profiling.

## Failure Triage

- A case is unexpectedly unsupported:
  inspect `supports_op` first, then tensor shapes in the test string.

- Stub passes but AOT fails correctness:
  compare the generated device source and launcher parameter order. Check packed
  layout math, dtype reinterpretation, and row strides.

- AOT fails to compile:
  fix generator post-processing, not generated files. Regenerate afterward.

- Counters are zero with `GGML_TILELANG_ENABLE=1`:
  check backend registration, runtime opt-in, buffer type support, and the exact
  backend name passed to `-b TileLang`.

- Counters are nonzero without `GGML_TILELANG_ENABLE`:
  stop and fix runtime gating before any benchmark.

## Milestone Rule

Every operator milestone should end with:

```text
stub path: correctness pass, counter > 0
AOT path:  correctness pass, counter > 0
disabled:  0/0 tests, counters = 0
benchmark: Release command recorded, baseline recorded
```

Only after that should the next task widen shapes, add a quantization variant,
or optimize performance.
