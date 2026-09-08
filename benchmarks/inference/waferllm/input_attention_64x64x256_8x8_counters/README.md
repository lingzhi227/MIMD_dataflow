# Shared-input resident attention chain — development

`hls.cpp` describes the source algorithm and dataflow using the public frontend. The experimental path uses the shared Clang frontend, structural verifier, 25-phase lifetime planner and composed CSL generation. Regular compiler dispatch deliberately rejects this unqualified profile.

The diagnostic driver is `experiments/input_attention_codegen_probe.py`. Default preparation refuses to proceed past failed numerical gates. `--diagnose-numerical-failure` explicitly prepares an unqualified SDK diagnostic, preserving the failed native gate. This option is not a qualification override.

Evidence paths and the precision limitation are recorded in PORT.json and `docs/RESIDENT-INPUT-PREFIX.md`. Do not copy these diagnostic results into a successful catalog registration. Three source-matched calls pass, while original case3 fails the fixed accuracy contract in actual native and SDK execution. All original inputs and failures remain frozen.
