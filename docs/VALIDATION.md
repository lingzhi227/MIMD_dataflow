# Validation and completion policy

For a bounded profile, completion requires supported frontend semantics, native execution, generated CSL compilation, completed SDK calls, target/state audits, independent numerical checks and the profile-specific source/protocol gates. Repeated calls and mutations test stale state and observer integrity where specified.

Do not collapse these levels:

1. Source exists.
2. Native checks pass and CSL is generated.
3. Official CSL compilation succeeds.
4. SDK simulator execution completes and passes its numerical/protocol checks.
5. The actual target model dimensions and real weights pass.
6. A complete application passes on real hardware.

The included profile index records historical level 4 evidence within each contract. It does not establish levels 5 or 6 for an LLM. The release's own CPU and packaging checks are recorded separately in `release/CHECKS.md`.

## Evidence interpretation

- Native C++ stdout, IR results, target arithmetic and independent mathematical references are separate evidence sources. Historical field naming is corrected in [NATIVE-EVIDENCE.md](../ports/docs/NATIVE-EVIDENCE.md).
- A source-control run is not automatically an HLS-generated run. Explicit source repairs and shared precision libraries are disclosed.
- `current_toolchain=false` means the historical snapshot differs from the captured development toolchain, not that the historical test failed. Packaging changes mean even `true` must not be interpreted as a new release SDK run.
- Maximum local PE cycles, global latency, host wall time and physical hardware throughput are different metrics.
- Static ELF allocation excludes dynamic stack highwater unless explicitly measured.
- A timeout with valid completed calls is still an incomplete qualification. Successful retries do not erase the first failure.
- Aggregate normwise accuracy is not a per-component relative-error guarantee.

The original full experiment archive remains outside Git. Compact reports retain original paths and hashes as provenance; they do not pretend that excluded files are present. See [selection policy](../release/SELECTION.md).
