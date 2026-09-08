# Command-line tools

- `hls_compile.py`: compile supported HLS into a fresh CSL execution bundle.
- `run_profiles.py`: run selected profile native checks and optionally the SDK simulator.
- `hls_debug.py`: inspect saved numerical and protocol observations.
- `update_status.py`: full-archive status maintenance; intentionally blocked in curated releases.

Run these from any directory with an explicit source/output path; profile selection resolves paths from the repository layout. `bootstrap.py` is internal source-checkout setup.
