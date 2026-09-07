# Implementation workspace

Start with the [repository overview](../README.md), [architecture](../docs/ARCHITECTURE.md) and [reproduction guide](../docs/REPRODUCING.md).

- `toolchain/include/`: supported C++ numerical interfaces and explicit spatial policies.
- `toolchain/`: parsing, typed IR, supported graph checks, resource planning, CSL emission and SDK transport.
- `toolchain/runtime/`: CSL computation, collective calls, routes, task state machines and reusable local kernels.
- `projects/<origin>/<profile>/hls.cpp`: executable frontend source.
- `projects/<origin>/<profile>/PORT.json`: exact algorithm, shape, precision and provenance contract.
- `catalog.json`: profiles selected by `run_ports.py`.
- [STATUS.md](STATUS.md): recorded per-profile SDK evidence, not a claim that every profile was rerun with the release toolchain.
- `docs/`: detailed contracts; historical numerical results retain their original measurement scope.
- `tests/`: portable unit and semantic regressions, including small historical witnesses.
- `experiments/`: research controls and SDK probes; some retain original lab-environment setup and need adaptation.

Do not run `define_ports.py` over an accumulated catalog: it is an initial authoring script that resets it. Builds use fresh run directories. Preserve failures and never substitute current implementation files into historical frozen bundles.

The original development chronology is in [the archive](../archive/DEVELOPMENT-LOG.md). The current release boundary is described in [the release status](../docs/STATUS.md).
