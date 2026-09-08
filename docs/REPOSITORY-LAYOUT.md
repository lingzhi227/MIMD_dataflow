# Repository organization and migration

## Reference designs

- [Cerebras SDK examples](https://github.com/Cerebras/sdk-examples) separate `tutorials` from `benchmarks`, with per-example source and run instructions. We separate guided learning from algorithm profiles and put source contracts beside each design.
- [MLIR's dialect development guide](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/) separates public interfaces, implementation and conversions. We use `include`, responsibility-based `lib`, `tools` and dedicated tests without claiming an MLIR implementation.
- [AMD/Xilinx Vitis HLS introductory examples](https://github.com/Xilinx/Vitis-HLS-Introductory-Examples) organize examples around features such as interfaces, pipelining and task-level parallelism. Our tutorial guide uses feature-based reading paths rather than exposing internal provenance directories as the only starting point.

## Old to new

| Previous location | Current location |
| --- | --- |
| `ports/toolchain/include/` | `include/pragma/` |
| `ports/toolchain/*.py` | `lib/{Frontend,IR,Analysis,Transforms,Conversion,Runtime,Numerics,Driver,Debug}/` |
| `ports/toolchain/runtime/*.csl` | `runtime/csl/` |
| `ports/toolchain/runtime/native.cpp` | `runtime/native/native.cpp` |
| `ports/run_ports.py` | `tools/run_profiles.py` |
| `ports/toolchain/compile.py` as a CLI | `tools/hls_compile.py` |
| `ports/toolchain/debug.py` as a CLI | `tools/hls_debug.py` |
| `ports/projects/<origin>/<profile>/` | `benchmarks/<domain>/<origin>/<profile>/` |
| `ports/projects/<origin>/upstream/` | `third_party/sources/<origin>/` |
| `ports/references/` | `third_party/references/` |
| `ports/tests/test_*.py` | `tests/unit/` |
| `ports/*fixtures.py` | `tests/support/` |
| `ports/tests/fixtures/` | `tests/fixtures/` (unchanged historical contents) |
| `ports/docs/` | `docs/contracts/` |
| `ports/evidence/` and `ports/STATUS.md` | `validation/evidence/` and `validation/STATUS.md` |
| Fresh run directories beside examples | `build/runs/` |

`hls-layout.json` contains the exact file migration map and stable profile paths. Python entry points configure a finite list of module roots; the compiler source files actually live in the new directories. No compatibility symlink or duplicate `ports` source tree is retained.

## Evidence and development boundaries

Historical JSON reports retain original paths and hashes. They refer to the execution archive, not new authoring locations. Raw fixtures and upstream reference content are not rewritten. The human-readable validation index links the relocated sources and reports.

The source refactor is implemented in the publication checkout. The separately running development workspace, remote canonical toolchain and immutable SDK runs are not moved while jobs are active. Future synchronization must apply the migration map and preserve the new layout; copying the old `ports/` tree over this repository would undo the refactor.

Fresh frozen bundles keep their established internal ABI. `lib/Support/source_tree.py` maps logical module/asset names to authoring paths when compiling, and resolves only local frozen assets when auditing a snapshot. This keeps directory organization independent from immutable execution provenance.
