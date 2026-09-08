# Resident input attention with explicit mixed precision

The public `hls.cpp` goes through the regular Clang frontend, checked typed IR,
planner, shared CSL backend and SDK driver. It expresses one resident chain:

`Xn=RMS(X,gamma); Q/K=rotate(Xn*Wq/k); V=Xn*Wv; P=softmax(Q*K^T/sqrt(64)); A=P*V; Z=A*Wo+X; Y=Z+MLP(RMS(Z,gamma))`.

This profile uses 64×64 activations, 256 hidden MLP features and an 8×8 PE mesh.
Its eleven public inputs are X, gamma, Q/K/V/O/up/gate/down weights and supplied
cosine/sine pairs. Both RMS operations share gamma. Pair order is `odd_even`.
This is a single-head unmasked numerical chain, not complete Prefill/Decode,
head replication, a causal mask, a cache update or an end-to-end model.

## Precision and dataflow

The original half variant is preserved beside this folder: a cancellation case
fails its unchanged original-input accuracy gate. Changing only block
accumulation does not fix it. Explicit operation types now retain V,
probabilities, PV, O and Z in f32. The second RMS computes in f32 and narrows to
half before the MLP. Three MLP products use half block partials and f32 block
merges. The final residual add computes in f32 and narrows to half.

Pragmas describe two-hop exchange, initial tile alignment, rotating-root score
reduction, double buffers, local DSR compute and row max/sum collectives. The
compiler verifies precision, descriptor widths, bounds and phase lifetimes;
Python transports original inputs and observations only. It does not calculate
intermediates between device stages.

The schedule explicitly owns colors 1–11, queues 3–7, local tasks 19/20/25/26
and microthreads 0–3. Compute uses DSR 1; row collectives use DSR 2 and matrix
communication uses DSRs 3–6, with phase-exclusive joins before reuse. These are
user reservations, not a claim to enumerate compiler/SDK temporary resources.
The linked resident ELF footprint is 35,952 bytes per PE; the remaining 13,200
bytes are static unallocated space, not a measured runtime stack allowance.

## Evidence and reproduction

The standard-driver bundle is `run-20260907T144456167691Z/`. It retains:

- Original source, Clang AST, frontend/checked/optimized IR, semantic graph and schedule.
- Native executable output plus thirteen separately executed native branch observers.
- Every generated CSL file, reusable runtime modules and frozen compiler implementation.
- SDK commands, logs, linked ELFs, raw half/f32 observations and execution qualification.

The current admission status is recorded in `PORT.json` and the generated
`../../../STATUS.md`; a successful native build alone is not SDK qualification.
From the ports root, the unified entry is:

```sh
python3 run_ports.py --select-exact waferllm/input_attention_mixed_64x64x256_8x8_counters
# On the configured SDK host, execute a fresh full batch:
HLS_CLANGXX=/usr/bin/clang++-17 .venv/bin/python run_ports.py \
  --select-exact waferllm/input_attention_mixed_64x64x256_8x8_counters \
  --sdk --sdk-threads 8 --sdk-suppress-trace --sdk-timeout 5400
```

The preflight replay is explicitly prior actual SDK evidence with byte-identical
CSL/source/fixtures. It is not a target-arithmetic prediction and cannot qualify
a new SDK run. Different programs must establish their own evidence.

`toolchain/debug.py <bundle> --check-completed --node p0_0 --epoch 3 --step 10`
shows the actual f32 Z observation and protocol state at the cancellation case.
The debugger uses frozen arithmetic; a partial completed-call review never
claims whole-run qualification. Counter mode does not retain the hidden tensor.

## Source and performance scope

The reference is WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`,
`Prefill/src/prefill.csl` and its communication library (Apache-2.0). The preserved
source control documents forward-only preshift repairs and precision changes;
it is not an untouched upstream binary. The precision-matched source engine and
HLS composition share f32 communication/math primitives. Their cycle comparison
measures composition and observer overhead, not independent primitive quality.
It excludes Python I/O and does not establish real-hardware throughput.

See `docs/RESIDENT-INPUT-PREFIX.md` and `docs/F32-PRECISION-BOUNDS.md` from the ports
root for retained failures, range assumptions and SDK math probes. In particular,
the wider exp probe failed the fixed accuracy gate; it was not silently accepted.
