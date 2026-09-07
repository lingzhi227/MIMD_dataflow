# Distributed SUMMA through HLS and native CSL collectives

This profile implements `C=A*B` on a square PE mesh. It follows the official
SDK `benchmarks/gemm-collectives_2d` algorithm at commit
`4866cf330333446cb5e529e10f36be4600d1df29`. Original Apache-2.0 notices remain
in the derived runtime templates. Upstream files are unchanged.

```cpp
#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,256>("a");
  auto b = spatial::input<256,128>("b");
  #pragma csl dataflow rows=8 cols=8 broadcast=rows_columns reduce=local fp=relaxed compute=vector
  auto result = spatial::matmul(a,b);
  spatial::output("result",result);
}
```

The experimental project pragma chooses a distributed schedule for a typed
matrix operation. Every PE holds column-major tiles of A, B and C. Round k
broadcasts A along rows from column k and B down columns from row k, then
accumulates their product into resident C. Global K order is increasing.
The two asynchronous SDK collective completions join through a blocked compute
task: one activates it and the other unblocks it. Compute blocks itself before
starting the next round. A separate entrypoint clears C and resets round state
on every invocation.

The compiler checks divisibility, mesh dimensions, DSD offset bounds, memory
budgets, input dependencies and the explicit relaxed floating policy. It emits
CSL using `<collectives_2d/pe>`, `<memcpy/memcpy>` and strided DSD/FMA arithmetic.
The resource plan records colors 0/1 and 4/5, task IDs 8–15, distinct x/y queues
and DSR ownership. The host packs inputs and reads results and diagnostics;
it does not compute intermediate panels or reductions.

## Executed evidence

`run-20260906T095126838453Z` passed SDK 2.10.1 / WSE-3 simulator execution for:

| Matrix dimensions M×K×N | PE mesh | Local arithmetic | Same-runtime calls |
|---|---|---|---|
| 64×64×64 | 4×4 | DSD vector | 4 |
| 64×64×64 | 4×4 | scalar comparison | 4 |
| 128×256×128 | 8×8 | DSD vector | 4 |

Each fresh bundle retains Clang AST, typed/optimized IR, resource schedule,
generated CSL, native executable, inputs, references, frozen toolchain,
SDK compiler command and results. Every PE exports C after every K-prefix and
timestamps around local multiplication. Audits regenerate CSL and the schedule,
check each prefix independently, and require exact final/history agreement.
The four inputs include random matrices, coordinate-coded routing witnesses,
zero-reset after a nonzero call, and nearest-even halfway witnesses.

Independent coordinator audits use `math.fsum`: the large case checks all
524,288 prefix values and 65,536 final values. See
`coordination/summa-16pe-review.json` and `coordination/summa-64pe-review.json`.

The numerical paths have different outcomes. Large-case native sequential f32
does **not** pass the original fixed `rtol=3e-5, atol=3e-6` accuracy screen.
The large-case generated CSL final and prefix values **do** pass that screen.
All paths pass the explicitly additional componentwise rounding contract in
[NUMERICAL-POLICY.md](NUMERICAL-POLICY.md). The native failure remains recorded;
a rounding bound does not replace an application's accuracy requirement.

## Performance scope

`evidence/summa-cycles-reduced-trace.json` compares identical inputs, mesh,
communication, diagnostics and runtime options. Vector local multiplication
takes 17,807 median cycles per PE/round versus 110,460 for scalar (6.203×).
Total simulator cycles are 1,208,560 versus 2,691,423 (2.227×). This measures
benefit from using CSL DSD/FMA in this schedule; it is not hardware speedup,
vendor BLAS parity, or proof of an optimal distributed implementation.

Runtime options are frozen: `suppress_trace=true`, `num_threads=16`,
`dump_core=true`. Exported histories and core diagnostics remain available.
The comparison rejects mismatched instrumentation. An older full-trace run,
`run-20260906T093212228413Z`, passed both 16 PE cases but timed out at 600s
on the large case after three saved calls. Its partial results and remote
traces remain preserved. The completed large run took approximately 519s
under a 900s watchdog. Timeout alone is not classified as deadlock.

## Replay and limits

Run `python run_ports.py --sdk --select mesh_gemm --sdk-suppress-trace
--sdk-timeout 900` in the configured workstation SDK environment. For a frozen
bundle, run its `implementation/validate.py` on that bundle. Use
`toolchain/debug.py BUNDLE --node p3_4 --epoch 0 --step 2` to inspect a PE's
K-prefix, expected and actual tile, and timing.

This is a bounded SUMMA slice, not complete BLAS or WaferLLM reproduction.
Alpha/beta, transposes, uneven tails, arbitrary graph composition, persistent
matrix handles, optional history-free deployment and hardware scaling remain
future work. Square meshes 2–8 and evenly divisible shapes are explicit profile
constraints, not general limits of CSL.
