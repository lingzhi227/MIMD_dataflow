# Reproduce the distributed FFT profiles

Run from the ports workspace. Python/NumPy and a C++17 Clang compiler are required for the native stage; SDK execution uses the pinned SDK2.10.1 environment. The SDK SIF hash and each frozen implementation are recorded in qualification evidence. The current development host is workstation; no wafer hardware execution is implied.

## Author and freeze a source profile

The authoring helper creates an explicit HLS source and a fresh build, runs the native C++ reference and freezes the frontend, semantic IR, schedule, resource plan, generated CSL, runtime options and execution driver:

```sh
python3 experiments/build_fft_profiles.py --n 32 --mesh 8 --direction forward --norm backward --instrumentation sampled
```

It preserves an existing source rather than overwriting a differently authored program. Add `--result-layout transposed_pencils` to declare the five-stage ownership policy. Direction and normalization are typed numerical semantics; result layout is a placement contract. Custom `--input-batches` values are original problem inputs, never expected outputs computed for the device.

The generated bundle path is printed. Copy the entire bundle to the matching remote workspace, then run its frozen driver there:

```sh
.venv/bin/python projects/sdk_examples/<profile>/<fresh-run>/sdk-execution-driver.py projects/sdk_examples/<profile>/<fresh-run> --timeout 3600
```

Each attempt must use a fresh bundle. Preserve failed compiler/runtime logs. Do not replace a frozen implementation or update a canonical remote toolchain while it is executing. For64³, the validated four-call sampled run used a7200-second host timeout; this bounds simulator wall time, not device latency.

## Catalog execution

After the FFT registration checkpoint, select a single complete key to avoid also matching an output-layout variant:

```sh
python3 run_ports.py --select-exact sdk_examples/fft3d_32_8x8_forward_backward --instrumentation sampled --sdk-threads 8 --sdk-suppress-trace
```

Add `--sdk --sdk-timeout 3600` inside the configured SDK host environment to compile and execute CSL. This creates a new six-fixture run, not a replay of an existing run directory. `--select` retains substring selection for deliberate groups. These fixture runs contain native stdout, independent direct-DFT checks, SDK results and protocol audits. Current native application evidence is explained in [NATIVE-EVIDENCE.md](NATIVE-EVIDENCE.md).

## Inspect and compare

```sh
python3 toolchain/debug.py <bundle> --node p7_7 --epoch 4 --step 15
python3 experiments/compare_fft_instrumentation.py <sampled-bundle> <counter-bundle> <fresh-comparison.json>
python3 experiments/compare_fft_layouts.py <restored-bundle> <transposed-bundle> <fresh-comparison.json>
python3 experiments/check_fft_roundtrip.py <forward-bundle> <inverse-bundle> <fresh-roundtrip.json>
```

The debugger selects one PE and local pencil, with real recorded endpoints where enabled. It does not reconstruct a full device trace. Comparison scripts first use each bundle's frozen auditor. The layout comparator performs an integer permutation of device bits only. Roundtrip inputs must be the exact forward-device outputs, including signed zero and subnormal values; no host FFT or rescale is allowed.

`fft_native_baseline.py` runs the unmodified SDK layout/RPC library on two matching inputs in a fresh evidence directory. It verifies the image hash, native module/output provenance and raw device bits. Source controls cover forward/backward normalization at16³,32³ and64³. All two-call comparisons are bit-identical;64³ sampled HLS/native max-local cycles are519498/519341 (0.0302%overhead). Other direction/norm modes have their own correctness and local-cycle records, not separate native-overhead controls.

Maximum-local device intervals exclude host gathering and are distinct from synchronized global latency, simulator wall time and real-hardware performance. Sampled and counter modes have different observation costs. See [DISTRIBUTED-FFT.md](DISTRIBUTED-FFT.md) for measured contracts and [FFT-DESIGN-REFERENCES.md](FFT-DESIGN-REFERENCES.md) for source relationships.
