# Hypersparse SpMV source recovery (not an implemented HLS profile)

Next dependency after QR: official SDK `benchmarks/spmv-hypersparse`, pinned
commit `4866cf330333446cb5e529e10f36be4600d1df29`. CSL references are unchanged
in upstream. Additional pinned Python files, extracted with git show, are in
`projects/sdk_examples/reference_host/spmv-hypersparse` with SHA256 provenance.

## Observed source contracts

- The top layout requires at least four PE rows. It allocates six SpMV colors,
  fourteen local-task entrypoints and two additional allreduce colors plus one
  entrypoint. Queue/DSR ownership must follow the library's explicit arrays,
  rather than applying a blanket reservation from a different mesh kernel.
- Matrix partitions contain sparse columns and a sorted compact row set.
  `mat_rows_buf` indexes positions in `y_rows_init_buf`, not raw matrix row
  offsets. Column index, location and length are separate u16 buffers.
- Inputs are distributed by column block, then by PE row. Original host input
  padding is one; padded entries must never be interpreted as real columns.
  Outputs use a separate row/column distribution and padding boundary.
- Matrix values and local input/output values are f32; indices and nonzero
  counts are u16. Host transport zero-extends u16 to SDK's u32 staging words.
- North/south trains distribute vector data and compute sparse local outputs;
  east/west trains merge/reduce sparse row/value packets. Buffers are reused
  across phases, so end-of-call storage is not automatically a local-product
  history. Termination and any diagnostic copy must respect those lifetimes.
- Official host loads Matrix Market through SciPy CSR, sorts indices and casts
  values to f32, then constructs sorted CSC. Its default benchmark option
  **replaces matrix values with random numbers**. An HLS operator must preserve
  its given values; that benchmark input policy is not part of SpMV semantics.
- `memory_usage.py` accounts for 6 bytes per local nonzero, 6 per sparse column,
  26 per sparse row, five input-vector buffers and a dense output buffer.
  Runtime/control reserves and actual compiler limits must also be checked.
- The supplied allreduce synchronizes reference clocks before measuring SpMV.
  This is a useful native timing facility; unlike earlier local factor timers,
  it can support a clock-adjusted interval when executed and audited correctly.

## Planned HLS contract work

Introduce integer index types and canonical sparse storage explicitly; do not
encode a masked dense matrix or silently round indices through a floating
semantic type. A first sorted, duplicate-free CSC profile can reject other
formats, while an explicit import/canonicalization adapter preserves the
original entry semantics. Empty rows, columns, partitions, explicit zeros,
uneven nonzero counts and boundary indices need dedicated execution tests.

Separate structural partitioning from device arithmetic. Capacity bounds are
part of the resource contract, checked per partition for every input before
SDK execution. The independent numerical oracle must read original sparse
entries, not only generated packed buffers. Keep unsupported duplicate or
ordering cases explicit; do not silently randomize values or reassign indices.

These notes are source research and implementation requirements, not a sparse
compiler/runtime or SDK success claim.

## Observed SDK2.10.1 incompatibility (2026-09-06)

Actual HLS512 compile `run-20260906T113521593969Z` failed because the original
input queue1 allocation collides with `<memcpy/wse3/memcpy>`'s RXCOMMAND.
The installed SDK source at its diagnostic path confirms input command queue1,
output command queue1 and the runtime H2D input queue0 configuration. This
supersedes any assumption that the original example's queue list works unchanged.
The adaptation moves RX_SOUTH to input2, retaining output2 in the separate output
bank; compilation and execution of that correction are still pending. Allreduce
keeps queue5, other SpMV receives4/6/7 and sends2/3 remain unchanged.
