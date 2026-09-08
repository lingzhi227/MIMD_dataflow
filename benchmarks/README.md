# HLS benchmark and application profiles

Applications are grouped by numerical domain, then upstream provenance. Every profile contains an `hls.cpp` and a `PORT.json` contract. A profile key remains stable even when its directory changes.

| Family | Contents |
| --- | --- |
| [linear_algebra](linear_algebra/) | Dense/sparse products, factorizations, reductions and iterative methods |
| [inference](inference/) | Normalization, activation, attention, MLP and supported resident combinations |
| [transforms](transforms/) | Local and distributed FFT profiles |
| [stencil](stencil/) | Bounded grid/stencil experiments; broader development is paused |
| [applications](applications/) | Limited application fragments; not complete production workflows |

`catalog.json` selects admitted runner profiles; `hls-layout.json` at the repository root maps stable keys to physical directories, including unqualified source candidates. Presence in this tree alone is not an SDK qualification. See [the captured validation index](../validation/STATUS.md).

```sh
python tools/run_profiles.py --select-exact waferllm/projected_cache_attention_3x256x512_8x8
```

Run from the repository root. Fresh builds go under `build/runs/`, and fresh reports under `build/reports/`. Upstream source is separate under `third_party/sources/`.
