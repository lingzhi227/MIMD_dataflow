# Pragma HLS development

Use the directory responsibilities and path map in `docs/REPOSITORY-LAYOUT.md` and `hls-layout.json`. Public commands are under `tools/`. Compiler implementation is under `lib/`; do not recreate the old `ports/` source tree.

Preserve `third_party/` references, `tests/fixtures/` historical witnesses and raw validation records. Never replace frozen run artifacts with current sources. Keep mathematical references separate from code generation and preserve explicit numerical/resource contracts. Changes to generation require native/semantic checks, generated-CSL comparison and appropriate SDK validation; report unavailable execution honestly.

The active development workspace and its remote SDK runs are separate from this publication checkout. Do not mutate their toolchain during running batches. Keep builds under `build/` and do not commit caches, SDK images or bulk traces.

The user-directed stopping condition remains: finish and audit existing category 8, report validation and unsupported scope, then stop. Do not begin categories 9–12, unrelated backfill or Qwen without further user instruction. Directory maintenance and this explicitly authorized GitHub publication do not expand algorithm scope.
