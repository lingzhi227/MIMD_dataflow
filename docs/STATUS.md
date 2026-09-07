# Release status

This source selection was prepared on 2026-09-07. The captured machine-readable index contains **122 bounded profiles**, each with a recorded `sdk_simulator` qualification. The latest included addition is the 64×64 projection/residual/RMS composition, registered in `qualification-20260907T074658560657Z.json`. Development continued while packaging; this is a fixed release selection, not a live dashboard.

## Completed within explicit contracts

- Distributed dense operations: GEMV, SUMMA, Cannon, Cholesky, no-pivot LU and QR.
- Sparse and reduction operations: hypersparse SpMV, dot, stable norm, resident CG, Jacobi-PCG, BiCGStab and fixed-step power method.
- SDK-backed distributed FFT and selected resident stencil configurations.
- Half two-hop matrix multiplication, grouped GEMV, RMSNorm, stable softmax, SiLU/gating, adjacent-pair rotation, normalized projection/fanout, scores and supplied-Q/K/V attention.
- Small ordinary-half gated MLP; two blocked mixed-precision MLP profiles with eight completed SDK calls each.
- 64×64 projection → residual add → RMSNorm, eight SDK calls, independent mathematical checks and adapted-source comparison.

All dimensions, input restrictions, observation modes and provenance are available in [the profile index](../ports/STATUS.md) and each `PORT.json`. Local/equation profiles must not be conflated with distributed counterparts.

## Failures and limitations retained

The ordinary-half 128×128→512→128 MLP failed the fixed mathematical accuracy contract, with about 5.44% error in the uniform case. It was not qualified. The explicit blocked-accumulation variants passed their original 2% L2 / 3% peak-scaled limits; this is not a blanket guarantee for model weights or arbitrary dimensions.

Initial blocked SDK attempts timed out after partially completed calls. Fresh bounded retries completed; the failed attempts remain documented. Source normalization experiments isolated stale descriptors and row-scale indexing defects; qualified comparisons identify the explicit repairs. Larger projection/residual/RMS configurations remain outside this release's qualified set.

No complete LLM, full WaferLLM prefill/decode reproduction, production inference service, or physical-wafer performance result is delivered here.

## Next model milestone

For Qwen2.5-0.5B-Instruct: freeze official weights/tokenizer/reference; prove real-shape memory placement; establish accumulation precision; implement Qwen half-split RoPE, GQA and causal attention; add persistent KV cache; validate a complete decoder layer; connect embedding, all 24 layers, final normalization, full-vocabulary LM head and device argmax; then validate a single physical wafer with batch 1 and a 2048-token total sequence budget.

Aggregate SRAM capacity is only a preliminary check. Per-PE data/code/stack, padding, replication, communication and scratch lifetimes require compilation and execution evidence.
