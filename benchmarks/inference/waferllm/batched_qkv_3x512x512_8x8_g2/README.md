# batched_qkv_3x512x512_8x8_g2

Source-backed normalized projection subgraph from WaferLLM Decode. See PORT.json for provenance and docs/DECODE-BATCHED-LINEAR-ALGEBRA.md for layout and protocol.

`hls.cpp` expresses one shared RMS and independent typed matmul branches with resident feature-sharded dataflow. Every fresh run keeps Clang AST, semantic IR, physical schedule/resource limits, generated CSL, native outputs, frozen compiler/runtime, and actual SDK evidence.

Eight standard SDK2.10.1 calls and eight precision/order-matched source calls passed. All three branches and all PE replicas/local witnesses match exactly. Maximum independent original-input relative L2 error is0.006321 and peak-scaled error0.007225. Max-PE cycle ratio is1.005604–1.005606; linked static high-water37168bytes/PE, excluding dynamic stack. All48 injected corruptions were rejected; actual C++ runs on both hosts pass and generate identical CSL.

Evidence: `run-20260907T170911843494Z`, `../../../evidence/batched-qkv-full8-source-comparison.json`, and `../../../evidence/qualification-20260907T182741945418Z.json`. This is one bounded configuration of a normalized projection subgraph, without a full Decode/cache or hardware-performance claim. Debug steps0–2 are RMS,3 is packed branch-local output,4 is packed reduced output. All branch boundaries and PE replicas are checked.
