# Tests and evidence fixtures

- `unit/`: frontend, semantic, numerical, lifetime, generation and audit regressions.
- `support/`: application input generation and independent numerical checks used by the runner and tests.
- `fixtures/`: selected historical witnesses. These files are immutable original evidence; path references inside them remain historical.
- `probes/`: explicit SDK resource and execution probes.

Run `python -m unittest discover -s tests/unit -v` from the repository root. Hardware/SDK availability is not implied by this CPU regression command. Source-generation comparisons and frozen-auditor tests cover important cross-layer behavior.
