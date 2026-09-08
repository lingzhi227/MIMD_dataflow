# Release validation — 2026-09-08

The update was prepared and checked in the isolated publication checkout, without changing active SDK runs or the developer's canonical toolchain.

- Captured profile consistency: all **141** HLS source hashes match the captured status; each corresponding SDK report case records a pass.
- First complete regression after fixture repairs: **332 tests passed** in 85.746 seconds. Final post-selection regression: **332 tests passed** in 85.725 seconds.
- Native `sdk_examples/gemm` smoke passed; see `native-smoke-20260908.json`.
- Root README, principal English guides, example tour and profile-index local links resolve.
- No credential-pattern matches were found for GitHub tokens, AWS access keys or private-key headers in the selected payload.
- No selected file exceeds 45 MiB. Roughly 331.5 MiB of unused historical data was removed after tracing fixture reads. Necessary regression witnesses remain separately identified under `tests/fixtures/history/`.

Two initial packaging failures were missing historical fixture dependencies, not numerical failures. The missing source-generation witness and complete manifest-bound frozen-auditor witness were restored without relaxing assertions or modifying original evidence bytes.

Validation environment: Python 3.14.3, NumPy 2.4.1 and the local Clang installation. This is an additional host check, not a claim that the pinned NumPy 2.2.6 environment was rerun during packaging. The reproduction guide recommends Python 3.10–3.13 for that pinned dependency.

No fresh SDK simulator or physical-wafer execution was performed for this GitHub update. Historical reports refer to original frozen runs. The current 35-node parent remains unqualified at the captured checkpoint. Full-catalog native reruns are not claimed by the release smoke.

The human-readable status table separates recorded SDK success from captured toolchain-version matching. The status regeneration script now refuses to reconstruct history from this intentionally incomplete archive.

Previous release checks are retained in `CHECKS-20260907.md`.
