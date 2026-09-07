# Native LU baseline reference

These are read-only reference copies of the previously validated SDK 2.10.1
single-RX migration, not HLS-generated files. Original upstream commit:
`016156e79b63fe45e118580da8db694285b6c6d9`.

`migration.patch` applies to the original repository root. The two CSL files
correspond to `lu_factorization/many_elements_per_pe`. Hashes of the copied
migration source are retained in `evidence/lu-migration-reference.json` at the
ports root. Prior evidence and discussion remain in the shared knowledge
repository's `reproduction/MATRIX_QR_LU_MANY_SDK2101_MIGRATION.md`.

Do not alter these baseline copies to make an HLS comparison pass. Changes to
the HLS runtime belong in toolchain/runtime; new baseline variants need fresh
reference and provenance records.
