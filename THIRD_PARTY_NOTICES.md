# Source attribution and licensing boundaries

Pragma combines original compiler/runtime work with attributed research and SDK-derived sources. Original notices, upstream license files, `PROVENANCE.json`, and per-profile `PORT.json` origin fields are retained. No blanket license is added over material with different terms.

- Cerebras SDK examples: attributed source trees under `third_party/sources/sdk_examples/`; retain their included notices and licenses.
- Matrix algorithm references: `third_party/sources/matrix_algorithms/` and the corresponding profile origins.
- WaferLLM: `third_party/sources/waferllm/` plus adapted runtime components. The Apache-2.0 text is retained in `runtime/csl/waferllm-LICENSE.txt`; source modifications and comparisons are described in the detailed contracts.
- Other numerical/stencil references: see each `third_party/sources/<origin>/` tree and `validation/evidence/source_inventory.json`.
- SDK math/FFT extracts: `third_party/references/` contains provenance and hashes tied to SDK 2.10.1. Their presence does not grant an SDK license. SDK images, toolchain binaries and credentials are not distributed here.

The original Pragma implementation has no newly assigned blanket open-source license in this update. Repository access does not by itself establish redistribution rights. This is an attribution statement, not a claim of upstream endorsement.
