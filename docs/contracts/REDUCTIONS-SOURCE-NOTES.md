# Next linear-algebra dependency: collective dot and stable norm

Source inspected: pinned SDK `benchmarks/conjugate-gradient/src/blas.csl`
and `kernel_cg.csl`, commit 4866cf330333446cb5e529e10f36be4600d1df29.
This preserves the initial design rationale. Implemented profiles, measured
results and current limitations are recorded in REDUCTIONS.md.

The SDK CG source computes local dot products through memory DSDs and `@map`,
then performs a distributed sum before updating the next recurrence. Its local
`nrm2` first finds a maximum magnitude, derives a power-of-two scale, sums scaled
squares, then applies sqrt and restores the scale. A direct unscaled sum of
squares is not an adequate replacement for a general norm.

The next HLS operators should expose distributed vector ownership and collective
result placement, with these layers separate:

1. Logical `dot`/`nrm2`, explicit f32 accuracy policy and supported input domain.
2. Vector partition, local reduction schedule, global collective and result
   ownership (root or replicated); padding contributes zero and must be checked.
3. SDK collective callbacks, queue/task/DSR lifetimes and resident state.
4. Host ABI, original-input independent arithmetic and local/global witnesses.

A distributed stable norm needs a global magnitude scale before each PE sums
scaled squares. Computing local norms and squaring them for a global sum can
reintroduce overflow. SDK exponent-based scaling is a reference; subnormal,
zero, finite-range and representable-result policies require actual CSL tests.

Reuse current SDK `collectives_2d` facilities, already exercised by GEMV/SUMMA,
where their supported operations match. Keep the unresolved legacy benchmark
clock-allreduce migration separate. Reduction completion must be observed before
scalar recurrence state is consumed. In future SpMV/collective composition,
local send completion is not a global phase barrier: color, queue and DSR reuse
requires an explicit lifetime argument or disjoint resources. Host relaunches
between every CG step do not establish a resident distributed solver.

Required experiments: repeated changed vectors; zero and cancellation cases;
non-divisible partition tails; independent full-vector dot/norm; all-PE local
partial checks and replicated-result checks; matched native SDK comparison.
For a vectorized local path, measure both local arithmetic and the complete
collective interval. Do not transfer the small-GEMV vector speedup claim to a
scalar dot reduction without measurement.
