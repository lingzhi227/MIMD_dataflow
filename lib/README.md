# Compiler implementation

The folders follow compiler responsibilities rather than application provenance. This is inspired by MLIR's separation of interfaces, implementation, transformations and tools; Pragma does not use or implement MLIR dialects.

| Directory | Responsibility | Representative modules |
| --- | --- | --- |
| Frontend | Parse supported C++ and explicit pragma syntax | `frontend.py`, `pragma_contracts.py`, `host_compiler.py` |
| IR | Typed operator/graph structures and structural validation | `ir.py`, `attention_ir.py`, `feed_forward_ir.py`, `projected_cache_ffn_ir.py` |
| Analysis | Shape/precision constraints, range bounds, resource/lifetime proofs | `region_lifetimes.py`, `mean_statistic_bounds.py`, `inference_resources.py` |
| Transforms | Supported scheduling and vectorization | `planner.py`, `vectorize.py`, graph-specific plans |
| Conversion | Verified graph lowering, CSL generation and composition hooks | `mesh_*.py`, `*_csl.py`, `projected_cache_ffn_codegen.py` |
| Runtime | Python SDK bindings, transport and ABI | `sdk.py`, `mesh_*_sdk.py`, `native_transport.py` |
| Numerics | Floating-point models and independent numerical references | `binary16.py`, `sdk_math_reference.py`, `*_reference.py` |
| Driver | Native build, frozen snapshots, integrity and execution audit | `compile.py`, `integrity.py`, `validate.py` |
| Debug | Saved device observation and phase inspection | `debug.py`, `*_debug.py`, `frozen_inspection.py` |
| Support | Resolve reorganized source files and immutable bundle assets | `source_tree.py` |

Use `tools/hls_compile.py`, `tools/run_profiles.py` and `tools/hls_debug.py` as public entry points. Individual implementation files are not the CLI.

Existing internal module names are preserved. `tools/bootstrap.py` configures explicit module directories from `hls-layout.json`; there are no duplicate source trees or symlinks. This refactor changes physical organization and path lookup, not algorithm policy. It is not a new installed Python package API.

Generated execution bundles deliberately retain the flat, self-contained module layout expected by their frozen loaders. Their files are copied from this source tree through a logical-name manifest. The bundle resolver uses only frozen assets when present and must never fall back to live authoring files. Historical frozen fixtures are kept byte-for-byte.
