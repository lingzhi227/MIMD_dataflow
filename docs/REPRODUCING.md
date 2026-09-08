# Reproducing the implementation

## CPU setup

Recommended reproducible dependency setup: Python 3.10–3.13, NumPy 2.2.6, and `clang++` supporting C++17 and `_Float16`. The SDK environment is a separate dependency and is not bundled.

Set `HLS_CLANGXX` to select a compatible native/frontend compiler, for example `export HLS_CLANGXX=/usr/bin/clang++-17` on the SDK host. The default is `clang++`; the actual command and version are recorded. Default Clang 14 on the development SDK host rejected `_Float16`, while the explicit Clang 17 path passed.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r ports/requirements.txt
python -m unittest discover -s ports/tests -v
cd ports
python run_ports.py --select-exact sdk_examples/gemm
```

The runner creates a fresh `projects/<origin>/<profile>/run-<UTC>/` directory. It parses the frontend, checks the supported numerical/dataflow contract, plans placement, emits CSL, compiles and runs native C++, and performs independent application checks. CPU success is not SDK execution success.

Choose exact keys from [the index](../ports/STATUS.md). `--select` performs substring matching; `--select-exact` selects one profile. A full native catalog run omits selection and can take substantially longer.

## SDK simulator execution

The recorded target is SDK 2.10.1 / WSE3. Install the SDK through your own Cerebras access. This release does not provide an SDK image or credentials. The original SDK bindings also pin an internal compiler prefix matching that image; other SDK versions require a separately validated port.

```sh
export PRAGMA_SDK_IMAGE=/absolute/path/to/sdk-2.10.1.sif
export PRAGMA_CS_PYTHON=/absolute/path/to/cs_python
cd ports  # from the repository root
python run_ports.py --sdk --sdk-suppress-trace   --select-exact sdk_examples/gemm --sdk-timeout 600
```

`PRAGMA_CS_PYTHON` must launch a Python environment containing the Cerebras SDK, NumPy and the source workspace mounts required by the SDK wrapper. The runner requires the recorded image SHA-256:

```text
fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d
```

The release replaces two author-specific host paths with these required environment variables and retains the SDK hash check. It does not claim that the release was freshly executed in SDK during packaging. The compact reports record previous executions of frozen development artifacts.

Use `--sdk-threads` to explicitly fix simulator workers where supported. Large examples may require much longer budgets; estimate from observed progress and launch fresh runs rather than changing an active run's timeout. Timeouts, compiler errors, simulator fatal errors and numerical failures are separate outcomes.

## Inspect a fresh run

```sh
python toolchain/debug.py projects/<origin>/<profile>/run-<UTC>   --node p3_2 --epoch 0
```

Supported phase/step arguments depend on the profile. Inspect final outputs, target arithmetic and independent application checks. Sampled and counter modes have different observation coverage; missing intermediate histories are not evidence of observed correctness.

## Historical evidence

Selected JSON reports and a few generated CSL outputs are included for review. Full frozen execution bundles, binaries, device dumps and bulk traces are excluded. Thus this checkout cannot re-audit every historical result offline. Generate a fresh run for execution reproduction; historical source bit-for-bit reproduction additionally requires the corresponding complete frozen bundle. See [the packaging policy](../release/SELECTION.md).

The `experiments/` directory contains research source controls and comparisons. Some still refer to the original lab paths or omitted runs. They are not the portable getting-started interface; inspect their setup before invoking them.

`status.py` requires complete historical run bundles and is disabled in this curated checkout to avoid replacing the captured index with incomplete results. Fresh run reports remain available in `ports/evidence/`.
