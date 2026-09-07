# Blocked neighbor-Givens QR, R-only

The HLS profile `mesh_qr.v1` preserves the Matrix algorithms blocked QR
schedule: adjacent local rows undergo Givens rotations, neighboring PEs exchange
rows, and sine/cosine coefficients travel east to the remaining column blocks.
This is an R-only interface. Q is not emitted by either this profile or the
referenced many-elements-per-PE kernel.

```cpp
#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,64>("a");
  #pragma csl dataflow rows=8 cols=4 exchange=neighbors rotation=givens fp=relaxed compute=vector
  auto result = spatial::qr_r(a);
  spatial::output("result",result);
}
```

The returned M×N upper trapezoid has unspecified signs on its nonzero rows.
Spatial scheduling may choose a different sequence of equivalent rotations
than the sequential adjacent-row C++ reference. Validation aligns one sign per
R row; it does not modify saved or returned device values. This is an explicit
factorization equivalence contract, not bitwise reproduction of the native
reference. Square local tiles of width at least two and tall/square matrices
are required. Current PE rectangles have 2–8 rows and 1–rows columns.

## Domain and numerical checks

Full column rank and `cond2(A)<=10000` are checked by the evaluator before
entering SDK execution. The native C++ and f32 reference also require a final
pivot margin greater than `2^-17*maxabs(A)`. Unsupported inputs are rejected;
there is no implicit regularization or column pivoting.

The independent checker verifies all R values, including its lower zeros,
against float64 QR up to row signs using fixed `rtol=3e-5, atol=3e-6`. It also
checks the relative infinity-norm Gram residual `R.T*R-A.T*A <= 3e-5`.
Within the conditioning domain, it derives Q via a linear solve with R's top
square block and requires `||Q.T*Q-I||inf <= 3e-5`. That Q is a host-derived
diagnostic, not observed device Q. Its reconstruction alone would not be an
independent correctness certificate, so Gram and factor checks are mandatory.

Inputs cover dense full-rank matrices, signed diagonal-after-dense, changed
scale, and reversed column order. Unit checks cover zero-b Givens, the
`abs(b)>abs(a)` branch, negative signs, rank deficiency, excessive conditioning,
incorrect factors and malformed rotation witnesses.

## CSL schedule, resources and repeatability

Original source commit: `016156e79b63fe45e118580da8db694285b6c6d9`,
`QR_factorization/many_elements_per_pe/{layout,pe_program}.csl`. The prior
SDK 2.10.1 migration uses separate north/south colors alternating by row parity
and restores single-input WSE-3 routes. It preserves neighbor sends and
Givens arithmetic. Reference copies and the migration patch are under
`experiments/reference/matrix-qr-sdk2101`; they are distinct from generated HLS
CSL and from the untouched upstream tree.

The generated runtime adds an all-PE prepare barrier that resets row and
auxiliary DSDs, fabric extents, counters, witness storage and horizontal
coefficient receive routes. Original block-local/neighbor Givens routines
then run using DSD vector moves, multiplies and FMAs. Colors 0/3 and 4/5 serve
vertical directions, color 2 carries horizontal coefficients; input queues
2/3/4 and output queues 5/6/7 leave SDK memcpy's queues separate.

The shared factor host binding supports rectangular PE arrays and typed
diagnostic shapes. It transports inputs/outputs, invokes prepare and starts
the kernel; all rotations and coefficients are computed in CSL.

## Sampled internal evidence

Each PE records its first seven rotations and a rolling nine-slot sample at
every sixteenth rotation. A witness contains role, serial, cosine, sine, two
pre-update values and the locally computed post-update value(s). Audits require
exact slot/serial/role matching, zero unused fields, approximately unit-norm
Givens coefficients and a componentwise three-rounding f32 arithmetic bound
for each observed rotation result. These are sampled local checks, not a
complete replay of all rotations or an independently emitted Q.

`qr_schedule.py` enumerates the original geometry-dependent control flow
without numerical data or device observations. Its expected total and ordered
roles are checked for every PE/epoch. This gives the first call an independent
count oracle; matching later calls alone would only demonstrate repeatability.
The 2×2 PE, width-two case has hand-checked totals `[[3,3],[4,5]]`.

## Executed configurations and instrumentation

Sampled mode completed four calls in one SDK runtime for 32×32/4×4 PEs
(`run-20260906T104749967279Z`), 128×64/8×4 and 128×128/8×8
(`run-20260906T105123777221Z`). Independent geometry and numerical reviews
are retained under `coordination/qr*-review.json`.

`--instrumentation counters` disables the rotation sampling call sites, keeps
exact per-PE rotation counts, and retains all final R/Gram/derived-Q numerical
checks. It does not offer the sampled mode's internal numerical observations.
The witness array remains allocated, zeroed during prepare and read by the host;
this storage/transport cost is not hidden. Sampled remains the default.

All three counters configurations passed four real SDK calls in
`run-20260906T111409529822Z`. `experiments/compare_qr_modes.py` re-audits each
bundle using its own frozen implementation in a temporary copy. All four f32
output bit patterns and every PE's rotation counts match the sampled runs
exactly. Saved source, input, runtime-option and native-baseline hashes are
checked; no baseline was redefined to make the comparison pass.

| Matrix / PE rectangle | Sampled / native | Counters / native |
| --- | ---: | ---: |
| 32×32 / 4×4 | 1.177724 | 1.005654 |
| 128×64 / 8×4 | 1.169756 | 1.005507 |
| 128×128 / 8×8 | 1.168207 | 1.005454 |

These ratios compare the maximum local PE factor interval for epoch 0 against
timestamp-only SDK2.10.1 migrated Matrix QR, with the same input and simulator
options. They exclude prepare and host I/O; they are neither synchronized global
latency nor hardware measurements. Every PE's individual interval is retained
in `evidence/qr{32,128x64,128}-instrumentation-comparison.json`. The original
roughly 17% sampled diagnostic overhead remains part of the evidence.

## Remaining scope

Native migrated-source timing comparisons must retain identical inputs and
matching per-PE factor boundaries. Sample instrumentation and host preparation
cost must remain visible in performance claims. The debugger supports `toolchain/debug.py BUNDLE --node p2_3 --epoch 0 --step 0`
for a rotation witness. Unretained serials are explicitly reported as not
retained, rather than filled with invented values. Complete Q generation,
arbitrary tile shapes, pivoted/rank-revealing QR, full rotation tracing and
composition into least-squares solves remain unimplemented.
