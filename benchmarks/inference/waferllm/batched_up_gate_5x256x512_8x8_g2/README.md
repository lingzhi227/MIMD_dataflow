# batched_up_gate_5x256x512_8x8_g2

Source-backed normalized projection subgraph from WaferLLM Decode. See PORT.json for provenance and docs/DECODE-BATCHED-LINEAR-ALGEBRA.md for layout and protocol.

`hls.cpp` expresses one shared RMS and independent typed matmul branches with resident feature-sharded dataflow. Every fresh run keeps Clang AST, semantic IR, physical schedule/resource limits, generated CSL, native outputs, frozen compiler/runtime, and actual SDK evidence.

Eight standard SDK2.10.1 calls and eight precision/order-matched source calls passed. Both branches and all PE replicas/local witnesses match exactly. Maximum independent original-input relative L2 error is0.006292 and peak-scaled error0.007560. Max-PE cycle ratio is1.007735–1.007740; linked static high-water20672bytes/PE, excluding dynamic stack. All44 injected corruptions were rejected. Actual C++ runs on both hosts pass and generate identical CSL. The frozen incremental watchdog passed each saved call and correctly distinguished full completion.

Evidence: `run-20260907T180940232501Z`, `../../../evidence/batched-upgate-full8-source-comparison.json`, and `../../../evidence/qualification-20260907T185614692044Z.json`. This is a normalized two-projection subgraph; gated activation/down projection/full Decode and hardware throughput are not included. Debug steps0–2 are RMS,3 is packed branch-local output,4 is packed reduced output. All branch boundaries and PE replicas are checked.
