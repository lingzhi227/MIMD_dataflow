# Cannon source schedule and migration plan

Reference: `projects/matrix_algorithms/upstream/Cannons_algorithm/{layout,pe_program}.csl`,
pinned Matrix-algorithms commit016156e79b63fe45e118580da8db694285b6c6d9.
The upstream README explicitly places initial alignment on the host. At PE(x,y),
initial blocks are A[y,(x+y)%P] and B[(x+y)%P,x]. Every round accumulates the local
block product, then shifts A left and B up, with wrap traffic routed across the
row or column. Round r therefore uses global block k=(x+y+r)%P, not the
increasing-global-K SUMMA order.

Six colors encode each axis's wrap path and two alternating nearest-neighbor
paths. Even coordinate PEs send before receiving; odd coordinate PEs receive
before sending. A and B exchanges are sequential and reuse one temporary tile.
Three matrix buffers rotate ownership through pointer and DSD-base swaps; the
fourth matrix buffer is C. The source local compute uses vector FMA over a row.
This is distinct from orthogonal root broadcasts and from WaferLLM's f16,
two-hop communication/computation overlap and explicit DSR schedule.

SDK2.10.1 migration must assign non-memcpy queues (WSE3 reserves0/1), initialize
each input/output queue's color, use modern fabric DSD queue fields and remove
the old RPC launch declaration. The layout must pass logical coordinates
explicitly for parity, avoiding assumptions about absolute fabric offsets.
Preserve the directed routes, wrap paths and synchronous exchange ordering.

Warm invocations must reset C, pointers, DSD bases, offsets and per-call counts.
The original program does not perform that reset; merely reusing its compute
function would accumulate old results and retain rotated input ownership.
Generated host packing is only a permutation, never a matrix multiplication.
Full original-input products, every round's cyclic-K partial result, sampled
received blocks, shift counts and PE timestamps will be checked separately.

Start with square64/4×4 and128/8×8, divisible even meshes and explicit relaxed
FMA policy. Reject odd mesh, non-square and memory-exceeding shapes initially.
Compare local vector FMA with a same-schedule scalar control and a source-based
SDK2.10.1 adapter; distinguish native reference roundoff from fixed accuracy.
No implementation or successful SDK execution is asserted by this plan.
