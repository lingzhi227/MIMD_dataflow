# Release validation

Validated on 2026-09-07 in the isolated publication checkout.

| Check | Result |
| --- | --- |
| Unit/semantic regression suite | 203 tests passed; 35.593 seconds |
| Native GEMM smoke | `sdk_examples/gemm` passed native execution, generation and application checks; see `native-smoke.json` |
| Captured profile consistency | All 122 HLS source hashes match the captured status; all 122 selected SDK report cases record `passed: true` |
| English entry documentation | Main README, guides and active Markdown contain no untranslated Chinese passages; original chronology is explicitly archived |
| Reader navigation | Root README, main guides and example-tour local links resolve |
| Credential-pattern scan | No GitHub tokens, AWS access-key patterns or private-key headers detected in the selected payload |
| Large-file check | No payload file exceeds 50 MiB; SDK images/binaries and bulk traces excluded |
| Missing SDK configuration | Runner exits with an explicit configuration error before starting SDK work |

Packaging initially exposed four tests depending on omitted historical files. Exact original witnesses and their notices were moved into small, hashed test fixtures; the complete 203-test suite then passed. Numerical tolerances and kernel algorithms were not changed.

No new SDK simulator or real-hardware execution was performed during release packaging. Included SDK reports describe previously completed frozen development runs. The portable runner changes were covered by CPU smoke and configuration-error checks, not a fresh SDK run.

The tested environment used Python 3.14.3 and the local Clang installation. NumPy dependency requirements are pinned in `ports/requirements.txt`. Full-catalog native and SDK reruns are not claimed by these release checks.
