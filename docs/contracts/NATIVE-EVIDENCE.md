# Native application evidence and historical scope

The build executes the C++ program and compares its output with typed IR. Historical run_ports reports then supplied IR reference.json values to the separate application oracle, despite labelling that field native_application_checks. Preserve those reports; do not retroactively claim direct stdout application coverage. Device checks and frozen numerical/protocol audits remain their own evidence.

The corrected runner checks actual native-output.txt and retains separate ir_application_checks. Fresh report `evidence/run-20260906T202157697783Z.json` passes79/79 applications through that path; native_application_source identifies the file. This is CPU evidence, not another SDK execution.

The executable prints f32 with nine significant digits. The shared decoder restores binary32 before exact comparisons, retaining signed zero and subnormals. u32 records remain integers. It rejects malformed epochs, duplicate ports, bad extents, non-finite/overflowing f32 and invalid u32. It does not loosen numerical tolerances.

Preserved report201844813894 has12 decimal-as-f64 application comparison failures; the fresh corrected report passes. `evidence/native-decimal-f32-decoding-failure.json` records the affected profiles and a concrete roundtrip value. Historical completed FFT stdout was also independently reviewed in `evidence/fft-native-stdout-application-review.json`; that review is read-only evidence, not a new execution.
