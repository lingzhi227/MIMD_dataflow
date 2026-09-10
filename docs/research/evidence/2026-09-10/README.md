# Frozen survey evidence — 10 September 2026

These are selected, unchanged records from the completed spatial-HLS survey experiment. The [bundle index](bundle-index.json) gives the SHA-256 and size of every included record. The original [build manifest](manifest.json) also names omitted files; it is preserved as provenance and is not a manifest of this smaller publication folder.

| Records | What they establish |
|---|---|
| [spatial-contracts.json](spatial-contracts.json) | Frozen protocol inputs, plan/source identities, finite interleaving outcomes, intentional negative witnesses and the unfolded mapping check |
| [spiral-four-point.json](spiral-four-point.json) | Exact four-point FFT formula and a fixed two-PE symbolic movement witness; no SPIRAL execution |
| [source.cpp](source.cpp), [pe.csl](pe.csl), [layout.csl](layout.csl) | Original restricted HLS input and the generated CSL for the measured profile |
| [batches.json](batches.json), [reference.json](reference.json), [results.json](results.json), [audit.json](audit.json) | Full selected inputs, reference values, simulator observations and numerical audit |
| [schedule.json](schedule.json), [semantic.json](semantic.json), [build configuration](build-configuration.json) | Exact bounded shape, policies and selected plan |
| [SDK command](sdk-command.json), [native command](native-command.json), [timing record](sdk-time.log), [simulator statistics](sim_stats.json), [SDK identity](sdk-image.sha256) | Compilation/run provenance and explicitly scoped timing/resource measurements |
| [trace-cleanup.json](trace-cleanup.json) | Deletion of 5,723,033,334 bytes of reproducible simulator traces after successful auditing |

The SDK 2.10.1 WSE-3 run used a 4×4 PE grid, 64×64×64 multiplication, and four small-integer input batches represented in binary32. All 16,384 final values and 65,536 saved intermediate observations agree for these inputs. The 130.35-second wall time includes compilation and initialization; 17,807 PE-local compute cycles exclude collective communication. Neither is an end-to-end hardware benchmark.

This folder is an evidence selection, not a runnable SDK build directory. It omits licensed SDK binaries, the native executable, duplicated implementation snapshots and simulator traces. Historical absolute paths in command records identify the original run. For a fresh run, use the repository's [build and SDK instructions](../../../REPRODUCING.md), selecting the desired registered profile and recording its environment. The [research note](../../spatial-contracts.md) explains the host-only checks and the abstraction boundary.
