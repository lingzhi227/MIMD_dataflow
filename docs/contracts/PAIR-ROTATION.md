# Pair rotation and explicit RoPE conventions

`rotate_pairs<pair_order::even_odd>(x, cosine, sine)` applies the usual
2D rotation to adjacent feature pairs. `pair_order::odd_even` first swaps the
two inputs, then applies that rotation. The input order is part of algorithm
semantics in C++, typed IR and CSL; it is not an optimization that can silently
change numerical meaning. The name describes a pair transform, not an entire
model's rotary embedding pipeline.

Cosine and sine tensors have shape `[M,N/2]` for token-dependent coefficients,
or `[1,N/2]` for coefficients broadcast over token rows. `coefficients=per_token`
or `coefficients=feature_pairs` must agree with the actual tensor shape.
Coefficient table generation, position offsets, head layout and interleaved
versus half-vector pairing are distinct model concerns; only adjacent pairs
and explicit supplied tables are currently supported. Inputs need not satisfy
cos²+sin²=1, so callers must provide trigonometric tables to request a rotation.

## Why the contract is explicit

Pinned WaferLLM Prefill `xq_rope` computes
`(odd*cos-even*sin, even*cos+odd*sin)`. In an isolated actual SDK2.10.1 probe,
zero-angle inputs swap every adjacent pair. This matches `odd_even` above,
and differs from identity on the supplied column order. It is not evidence
about every possible model-specific upstream packing convention.

The source allocates temporary vectors and DSDs with length `local_features/2`,
while input/output column descriptors have length `local_token_rows`. Actual
8×16 local execution matches its pair formula. At8×32, the mismatched lengths
produce120/120/240 differing half words across three fixtures. Changing only
temporary storage/DSD lengths to8 restores all source-formula results exactly.
The generated runtime sizes every participating vector descriptor by token
rows. It does not reproduce the invalid shape-dependent behavior.

See `rope-source-pair-and-dsd-review.json` and
`rope-source-row-scratch-metadata-correction.json`. The earlier corrected-run
review's numerical data was right but its descriptor-length label remained16;
the correction explicitly reports actual8 and original16. Both are preserved.
Two earlier probe wrappers failed compilation due to mutable global DSD
initialization; their failures were wrapper errors, not source-kernel failures.

## CSL lowering

`mesh_pair_rotation.v1` partitions complete adjacent feature pairs into
column-major PE tiles. Pairs cannot straddle PEs. Each PE computes four half
products on token-length memory DSDs, followed by half subtract/add. Broadcast
coefficients use scalar operands; token-dependent coefficients use memory DSDs.
Inputs remain immutable, output is separate, and four token vectors are reused
for each pair. Sampled mode saves the four product vectors before reuse;
counter mode retains completion and final-result checks only.

There are no application fabric colors, queues, local tasks or microthreads in
this pointwise phase; SDK memcpy owns launch and host I/O. No explicit DSR
allocation is needed. Memory admissibility includes all inputs, output,
coefficient ownership, row scratch, observation storage and code/stack reserve.
Linked static sections and actual execution are separate gates from estimates.

## Accuracy and qualification scope

Native C++ evaluates the supplied pair formula in double then rounds its
output to half. The device target oracle follows four rounded half products
and rounded subtract/add. Their difference is checked per output against
`.0015*(abs(product0)+abs(product1))+2^-23`. This avoids pretending cancellation
has a small output-relative error. Inputs are finite exact half values of
magnitude≤8; product and sum overflow is excluded by this bound.

Six fixtures include identity angles, quarter-turn angles, token-dependent
angles, zero input, cancellation and changed inputs. Exact product witnesses,
immutable inputs, output bits, pair/epoch completion and timestamps are audited.
The stage debugger labels sampled versus unobserved data and tile ownership.
Source controls preserve the original arithmetic/DSD loop, explicitly repair
row scratch, and match optional product observations. Source input is in-place;
HLS output is separate. Any reported local simulator ratio includes those
storage and progress differences. No complete model or hardware claim follows.

As of the initial implementation,64×128 token-dependent and source-order
broadcast HLS bundles are executing.128×1024 is being prepared. These are
not yet registered as qualified algorithm profiles.

Both64×128 profiles subsequently passed all six calls and12corruption checks,
and were registered by `qualification-20260907T004436294008Z.json`. Three
matched source calls include nontrivial coefficients; all product/final bits
are exact. Broadcast HLS/source local intervals1131/1612cycles include their
storage/progress differences. Linked static high water: token6432B,
broadcast6144B.128×1024 remains a separate executing profile.

## Large-profile qualification update

128×1024/8×8 per-token even_odd now completes six SDK calls and12mutation rejections. Registration: `qualification-20260907T011926896987Z.json`; linked static25696B. No larger-geometry source performance ratio is claimed.
