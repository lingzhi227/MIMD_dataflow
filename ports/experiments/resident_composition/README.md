# Resident composition dependency probe

This experiment first creates a fresh ordinary HLS SpMV build, with native C++
and semantic checks, and keeps it unchanged under `hls/`. A separate `device/`
adapter imports the same runtime libraries and the installed SDK collectives.
It is an explicit composition experiment, not a completed CG frontend lowering.

One host launch performs a PE ownership transpose, the inverse transpose, SpMV,
and a global sum of squared output values. Four changed-input calls use distinct
index-coded vectors, zero, alternating signs and reversed indices. Audits check
both permutation and roundtrip bits, all original sparse entries through the
independent numerical reference, partials and all-PE cumulative progress.
The latest harness also exports the raw queue-empty masks at all three phase
boundaries. It checks all non-memcpy queues, excluding0/1. Empty queues are a
necessary local condition, not proof of arbitrary peer-wide quiescence.

The unchanged SDK collective imports own both queue banks at3/6 and5/7.
The enclosing adapter initializes the remaining SpMV queues. SDK `tile_config`
helpers switch input/output color associations after callbacks and empty checks;
CCL COLOR_0 uses its installed scatter filter and SpMV disables filtering.
SpMV has13 local tasks, CCL has4 and the phase dispatcher has1. Static task
availability is not a sufficient resource legality test.

Run on the SDK host:

```
.venv/bin/python experiments/resident_composition/run.py
.venv/bin/python experiments/resident_composition/run.py --large
```

The first size is512²/16PE; `--large` is4096²/64PE. No host numerical iteration
is inserted into the four-stage device call. No performance benefit is claimed
for doing a deliberately redundant transpose roundtrip: it is a routing test.

Preserved early failures:133702401920 (second layout block),133803273116
(actual duplicate input initialization, exposing an incorrect queue ledger),
134246027102 (module import inside function),134355224566 (member access on a
returned runtime struct requires a temporary). These are dated2026-09-06.
The first executing configuration134525092553 finished four calls but its report
writer missed a manifest argument. Fresh134727430627 passes the complete audit.

The generated adapter manifest records its sources separately from the original
HLS manifest. Its audit substitutes adapter source hashes for the standalone
CSL-regeneration check while retaining the original numerical/storage/progress
checks. Source-authoring and numerical oracles stay outside the compiler.

Large default-capacity134924963979 and compact608/384/384140001193248 fail at
link for SRAM. `--large --compact --capacity 593,364,362` passes in140648455668:
same original matrices, narrower explicit capacity contract,32-element transpose
chunks borrowing inactive SpMV north-partial storage. Only320 bytes remain below
48KiB static high-water; this is not measured runtime stack headroom. See
`docs/RESIDENT-SOLVER-RESOURCES.md` for callback guarantees and proof boundaries.
