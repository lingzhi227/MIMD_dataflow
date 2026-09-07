# Pragma: spatial HLS for Cerebras CSL

Pragma is an experimental C++-to-CSL toolchain for explicit numerical dataflow on Cerebras wafer-scale processors. It combines typed tensor operations and spatial pragmas with reusable CSL compute and communication libraries. The official Cerebras compiler and SDK compile and execute the generated programs.

**This repository contains a curated research implementation, not a production LLM runtime or a replacement for the Cerebras SDK.** The release snapshot includes **122 bounded profiles with recorded SDK simulator validation**. A profile specifies dimensions, precision, input domain, placement and observation mode; several profiles may implement variants of the same algorithm. Simulator evidence does not establish hardware throughput.

## Start here

- [What works and what remains](docs/STATUS.md)
- [Architecture and source map](docs/ARCHITECTURE.md)
- [Install, run and reproduce](docs/REPRODUCING.md)
- [Annotated example tour](examples/README.md)
- [Per-profile validation index](ports/STATUS.md)
- [Evidence policy and release checks](docs/VALIDATION.md)
- [Source attribution and licensing boundaries](THIRD_PARTY_NOTICES.md)

## What is implemented

| Area | Implemented and evaluated capabilities | Important boundary |
| --- | --- | --- |
| Dense linear algebra | Distributed GEMV, SUMMA, Cannon, Cholesky, no-pivot LU, QR; half two-hop GEMM and grouped GEMV | Algorithm domains and supported mesh/shape configurations are explicit |
| Sparse and iterative methods | Hypersparse SpMV, collective dot/norm, resident CG, Jacobi-PCG, BiCGStab and fixed-step power iteration | Bounded matrices, storage contracts and termination policies |
| Transforms and grids | Distributed SDK-backed FFT; resident stencil iterations | Supported layouts/boundaries only; no general physical simulator |
| Inference building blocks | RMSNorm, stable softmax, SiLU/gating, pair rotation, normalized projections, supplied-Q/K/V attention, resident MLP, projection/residual/RMS composition | No complete prefill/decode model pipeline; current pair rotation is not automatically Qwen RoPE |
| Tooling | Typed frontend, numerical/resource checks, CSL generation, SDK transport, frozen builds, state inspection and independent numerical checks | Explicit supported lowerings, not unrestricted graph compilation |

## Repository layout

```text
.
├── docs/                       # English overview, status, reproduction and evidence policy
├── examples/                   # Reading guide and selected actual generated CSL
├── ports/
│   ├── toolchain/              # Frontend, IR, resource planning, code generation and SDK bindings
│   │   ├── include/            # C++ tensor/operator interfaces
│   │   └── runtime/            # Reusable CSL math, routes and communication implementations
│   ├── projects/               # hls.cpp + PORT.json profiles, grouped by upstream provenance
│   ├── tests/                  # Semantic, numerical, resource and regression tests
│   ├── experiments/            # Source controls, compiler probes and comparison tools
│   ├── docs/                   # Detailed English algorithm and resource contracts
│   ├── evidence/               # Selected recorded qualification/failure reports
│   ├── references/             # Small pinned reference extracts and provenance
│   ├── catalog.json            # Profile definitions
│   └── run_ports.py            # Fresh native/SDK build and validation entry point
├── release/                    # File hashes, packaging checks and selection policy
└── archive/                    # Original mixed-language development log, for historical context
```

The `ports/` layout is retained deliberately: compiler snapshots, test fixtures and profile tooling use these relative paths. Earlier prototype compilers, SDK images, credentials, duplicate run snapshots and bulk device traces are not part of this source release.

## Quick start: CPU validation

Use Python 3.10+ and Clang with C++17 and `_Float16` support. SDK execution additionally needs the separately installed, pinned SDK 2.10.1 environment.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r ports/requirements.txt
python -m unittest discover -s ports/tests -v
cd ports
python run_ports.py --select-exact sdk_examples/gemm
```

This command compiles and executes native C++, generates CSL, and checks the numerical reference. It **does not execute CSL in the simulator** unless `--sdk` is supplied in the configured SDK environment. See [reproduction](docs/REPRODUCING.md) for that distinction and setup.

## Direction

The next model-driven goal is a complete single-wafer Qwen2.5-0.5B-Instruct inference path: real weights, all 24 layers, full vocabulary, prefill and KV-cache decode. Missing capabilities include model-correct GQA/causal attention, persistent KV state, embedding, vocabulary-wide output selection and full-model placement. This is a roadmap, not a delivered model implementation.

This repository update replaces the previous working tree through a normal commit. Earlier contents remain available in Git history. No Cerebras or upstream-project endorsement is claimed.
