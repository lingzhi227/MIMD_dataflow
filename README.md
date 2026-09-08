# Pragma HLS

A research C++-to-CSL compiler for explicit numerical dataflow on Cerebras wafer-scale processors. Write typed tensor operations and spatial policies in C++; Pragma checks and lowers supported graphs into CSL, which the official Cerebras compiler and SDK execute.

**141 bounded profiles have recorded SDK simulator qualifications.** The current 35-node attention-plus-FFN composition is still a development candidate in this snapshot. This is not a production LLM runtime or a replacement for CSL. [Read the exact status](docs/STATUS.md).

## Find what you need

| I want to… | Start here |
| --- | --- |
| Understand the project | [Architecture](docs/ARCHITECTURE.md) and [repository layout](docs/REPOSITORY-LAYOUT.md) |
| Learn the HLS interface | [Tutorials](examples/tutorials/README.md) and [public C++ headers](include/pragma/README.md) |
| Run an algorithm | [Benchmark families](benchmarks/README.md) |
| Work on the compiler | [Compiler source map](lib/README.md) |
| Inspect reusable CSL | [CSL runtime library](runtime/README.md) |
| Reproduce validation | [Reproduction guide](docs/REPRODUCING.md) and [qualification index](validation/STATUS.md) |
| Review scope and provenance | [Evidence policy](docs/VALIDATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md) |

## Layout

```text
include/pragma/          Public C++ HLS interfaces
lib/                    Compiler implementation, separated by responsibility
  Frontend/             Clang parsing and pragma syntax
  IR/                   Typed graph structure and semantic checks
  Analysis/             Numerical ranges, resources and lifetimes
  Transforms/           Scheduling and explicit transformations
  Conversion/           Supported graph lowering and CSL emission
  Runtime/              Python SDK bindings and transport
  Numerics/             Precision semantics and numerical references
  Driver/               Build orchestration, integrity and validation
  Debug/                Execution-state inspection
  Support/              Source-layout and frozen-bundle resolution
runtime/                CSL kernels/communication and native C++ support
  csl/
  native/
tools/                  Compile, run-profile and inspect entry points
examples/               Tutorials and selected generated CSL for reading
benchmarks/             HLS applications grouped by algorithm domain
  linear_algebra/
  inference/
  transforms/
  stencil/
  applications/
tests/                  Unit tests, numerical fixtures and SDK probes
third_party/            Preserved upstream sources and reference extracts
docs/                   Guides, architecture and detailed contracts
validation/             Captured qualification index and historical reports
experiments/            Source comparisons and research probes
scripts/authoring/      Profile authoring utilities
archive/                Original development chronology
release/                Release selection, migration map and validation records
build/                  Fresh generated runs and reports (ignored by Git)
```

## Run from the repository root

A pinned dependency setup uses Python 3.10–3.13 and NumPy 2.2.6. Clang must support C++17 and `_Float16`. Set `HLS_CLANGXX` when the default compiler is unsuitable.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests/unit -v
python tools/run_profiles.py --select-exact waferllm/mlp_128x128x512_8x8_blocked
```

This executes native checks and generates a frozen CSL bundle. SDK execution is a separate step requiring your configured SDK 2.10.1 installation and `--sdk`; see [reproduction](docs/REPRODUCING.md). A recorded historical SDK pass does not mean every example was rerun with this reorganized checkout.

## Current development boundary

Finish and audit category 8—WaferLLM numerical stages and accepted prefill/decode composition—then stop and report results and unsupported scope. Categories 9–12, unrelated applications and Qwen development require new user instructions. A single passing graph does not close category 8.

The original development workspace and active SDK runs are separate from this refactored publication checkout. Their immutable evidence has not been rewritten. No upstream endorsement is claimed.
