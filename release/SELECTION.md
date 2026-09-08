# Source selection and packaging policy

This release selects the active `csl-hls/ports` implementation rather than copying earlier prototype trees or the entire experiment workspace. The selected implementation is now organized by compiler responsibility, runtime, tools and benchmark domain. `hls-layout.json` records old-to-new paths; frozen execution bundles retain their established internal structure.

Included: compiler/frontend source, public C++ headers, CSL runtime libraries, profile `hls.cpp` and `PORT.json` files, small attributed upstream source trees, numerical fixtures, tests, research comparison tools, detailed algorithm contracts, the captured status/catalog, selected qualification/failure reports and three representative generated CSL outputs.

Excluded: earlier compiler prototypes, SDK images and executables, credentials, build caches, full repeated run snapshots, binary device artifacts and simulator traces. Selected historical tensor results required by regression tests are explicitly included under tests/fixtures/history. Full original runs and failures remain in the development workspace. Selected reports may name excluded paths. No omitted artifact is silently regenerated and presented as original evidence.

English README, architecture, status, reproduction and validation pages are the supported reader entry points. Most detailed contracts were already English. The original mixed-language development log is isolated under `archive/` for provenance, not used as the current user guide. Historical contract measurements remain dated and may describe earlier milestones; the release status takes precedence for aggregate counts.

Release-only edits: source-directory refactoring and explicit asset/import resolution; English documentation and navigation; relocations of small historical test witnesses into `tests/fixtures/`; explicit SDK image/wrapper environment variables in the runner; generated release manifests and checks. Kernel algorithms and numerical tolerances are not changed for packaging.

`MANIFEST.json` hashes all tracked payload files other than itself. `verify_manifest.py` checks their integrity. These are release payload hashes, not a replacement for original device execution manifests. The prior GitHub tree is preserved as the parent commit of the replacement.


September 8 update: test-dependent historical witnesses now live under `tests/fixtures/history/`. This is a selected regression dataset, including some frozen implementations and numerical results required by tests, rather than the full development archive. Original path/hash mappings are retained in `HISTORICAL-WITNESSES.json`; fixtures are unchanged bytes. Existing generated examples remain provenance-bound historical examples, not silently regenerated current outputs.

One frozen-auditor regression retains its manifest-bound native CPU executables as hash witnesses. The test verifies their bytes; it does not execute those historical binaries. Fresh native tests compile their own executables. SDK binaries remain excluded.
