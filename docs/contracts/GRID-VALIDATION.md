# Resident grid validation and performance evidence

The final historical grid batch, `run-20260906T081445392252Z`, validated five configurations using native C++, f32 IR, independent NumPy and SDK 2.10.1 / WSE3 simulation: 1×1×7 for 3 steps; 2×2×32 for 4; 4×4×128 for 16; 8×8×128 for 16; and a same-size scalar control. Each used two inputs.

The largest case uses 64 compute PEs, 8,192 field values and 16 resident steps. Across both inputs it audits 16,384 final values, 262,656 timestep values and control counts. Intermediate steps execute on-device. External boundaries are zero and coefficients are shared. Arithmetic order differs from the SDK native FMA implementation; upstream bitwise equivalence is not claimed.

## Controlled compute comparison

[Vector/scalar report](../../validation/evidence/grid-vector-scalar-cycles.json): 2×2×32, four steps and two inputs, with the same DSD communication and I/O. Scalar execution measured 55,603 simulator cycles and vector execution 28,377, a ratio of about 1.96. Every timestep had identical bits. This is an internal transformation comparison including I/O and history capture, not speedup over a Cerebras hand-written kernel.

## Runtime evolution

[Runtime report](../../validation/evidence/grid-runtime-cycle-evolution.json): 4×4×128, 16 steps and two inputs, all timestep bits identical.

| Version | Simulator cycles | I/O tiles | Fabric |
| --- | ---: | ---: | --- |
| Per-PE I/O, per-wavelet receive | 1,340,685 | 32 | 14×32 |
| Column I/O, DSD memory copies | 975,565 | 8 | 14×14 |
| Full-frame DSD receive, explicit microthreads | 393,458 | 8 | 14×14 |

The first/last ratio is about 3.41. Multiple mechanisms changed, so it cannot be attributed to a single compiler pass. Host wall time is not chip throughput.

## Failures and remaining scope

The older 64-PE per-PE I/O case exceeded 240 seconds. A later layout and budget completed. Asynchronous DSD communication compiled but failed at runtime without explicit microthread allocation; corrected allocation passed. See [CSL experiments](CSL-experiments.md).

SDK color-overlap warnings remain in the 64-PE historical logs. Successful tests are scoped observations, not a proof for every timing and layout. Real hardware measurements, full matched SDK blocked/DSR/FMA overlap comparisons, multiple fields, periodic boundaries and complete upstream applications remain outside this result. Full raw run bundles are not included in this source release.
