# Build, run and inspect

All commands below run from the repository root. A pinned dependency setup uses Python 3.10–3.13 with NumPy 2.2.6. Clang must support C++17 and `_Float16`; select another executable with `HLS_CLANGXX` if necessary. The SDK host's Clang 17 path was validated during development; default Clang 14 rejected half types there.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests/unit -v
python tools/run_profiles.py --select-exact waferllm/mlp_128x128x512_8x8_blocked
```

Profile keys are listed in [the qualification index](../validation/STATUS.md). `--select-exact` chooses one; `--select` is substring matching. Without a selection, the runner attempts the full catalog and may take substantially longer.

The runner executes native C++, builds and checks typed IR, plans resources, emits CSL and verifies its frozen snapshot. Generated runs and reports go to `build/runs/` and `build/reports/`, respectively. Native success does not imply SDK execution.

## SDK execution

Install SDK 2.10.1 separately through your Cerebras access and configure the image and wrapper paths:

```sh
export PRAGMA_SDK_IMAGE=/absolute/path/to/sdk-2.10.1.sif
export PRAGMA_CS_PYTHON=/absolute/path/to/cs_python
python tools/run_profiles.py --sdk --sdk-suppress-trace   --select-exact waferllm/mlp_128x128x512_8x8_blocked --sdk-timeout 5400
```

The runner retains the pinned SDK image hash check. The wrapper must provide the SDK Python environment and workspace mounts. SDK versions, internal compiler prefixes, worker count, timeout and instrumentation affect reproducibility; other environments need separately recorded validation. SDK images, tools and credentials are not distributed here.

## Direct compilation and inspection

```sh
python tools/hls_compile.py path/to/hls.cpp -o build/my-fresh-run
python tools/hls_debug.py path/to/generated/run --node p3_2 --epoch 0
```

The output directory must be fresh. Direct compilation defaults are suitable only for designs matching the supplied epochs, bounds, partitions and fixtures; the profile runner supplies the registered configuration. The inspection interface and observed stages depend on the graph and instrumentation mode.

## Historical records

Historical qualifications are indexed in `validation/`. Selected frozen regression witnesses live in `tests/fixtures/`; the full experiment archive is not included. `tools/update_status.py` refuses to rebuild a historical index from this incomplete curated archive. It must not overwrite prior SDK success with a partial reconstruction.

Read [the layout/migration guide](REPOSITORY-LAYOUT.md), [validation policy](VALIDATION.md) and [release checks](../release/CHECKS.md) before interpreting current native checks as historical SDK reproduction. Research controls in `experiments/` may still require original lab artifacts and environment configuration.
