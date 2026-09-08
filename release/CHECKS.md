# Directory refactor validation — 2026-09-08

The refactor was prepared in the isolated publication checkout. Active development trees and SDK experiments were not moved or modified.

- Full regression: **338 tests passed** in 99.798 seconds, including six repository-layout tests. After the final candidate relocation into the inference domain, **15 relevant tests passed** in 7.191 seconds.
- Migration audit: **4,168 moved files have identical Git blob hashes** against the pre-refactor commit. This covers existing public headers, CSL/native runtime, profile sources/contracts, upstream reference files, raw reports and historical fixtures. See [MIGRATION-CHECKS.json](MIGRATION-CHECKS.json).
- Blocked MLP native execution and profile checks passed. Its six generated CSL files match the selected qualified historical bundle.
- The 25-node projected-cache composition passed native execution and profile checks. All nine generated CSL files match a fresh run of the **same pre-refactor GitHub revision**, establishing that the directory move did not change generated CSL.
- The 35-node projected-cache + FFN candidate passed native compilation, frozen code-generation checks and the native/target preparation gates. All twelve generated CSL files match its selected frozen candidate. Target preparation is not SDK simulator execution.
- Public commands work outside the repository working directory; frozen resolvers stay isolated from live compiler files; standalone numerical reference modules remain independently importable.

[REFACTOR-CODEGEN.json](REFACTOR-CODEGEN.json) also records a comparison against the older SDK-qualified projected-cache bundle. One RMS library file differs there due to an optional mean-mode feature that already existed before this refactor. That historical difference is retained rather than described as a byte-identical match. The matched-revision comparison above passes for every generated CSL file.

No fresh SDK simulator or physical-wafer execution was performed for this refactor. The captured status remains **141 qualified bounded profiles**; the 35-node candidate remains pending SDK qualification. Neither full-catalog native reruns nor complete-model support is claimed.

Validation used Python 3.14.3, NumPy 2.4.1 and local Clang. The pinned SDK/reproduction environment is documented separately in [REPRODUCING.md](../docs/REPRODUCING.md). Fresh build outputs remain ignored under `build/`; the release manifest covers the published payload, not those local execution bundles.

The preceding source-update checks are preserved in [CHECKS-20260908-SNAPSHOT.md](CHECKS-20260908-SNAPSHOT.md). The organizational rationale and complete migration-map entry point are in [REPOSITORY-LAYOUT.md](../docs/REPOSITORY-LAYOUT.md).
