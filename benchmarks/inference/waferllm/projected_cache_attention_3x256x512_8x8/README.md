# Normalized QKV and supplied-cache attention

This HLS graph composes RMS normalization, three Q/K/V projections, explicit
adjacent-pair transforms, stable shared-cache attention, output projection and
original-input residual on an8×8PE mesh. It uses B=3,N=256,S=512, resident matrices
and cache shards, SDK X/Y collectives and local half DSR computation. The old cache
is read-only. New rotated K and projected V are separate public outputs.

`SOURCE-MAP.json` links all eleven mathematical stages to exact pinned Decode
function bodies, line locations and hashes, including explicit repairs and scope.

`hls.cpp` is the algorithm/dataflow source. `PORT.json` records upstream origin
and the supported contract. Each `run-*` directory retains its frontend source,
typed semantic IR, schedule/resource/lifetime plan, native program and stdout,
independent stage gates, generated CSL, frozen implementation and SDK evidence.
Failures are retained as part of the debugging record.

Qualified run: `run-20260907T235739454777Z`. Eight actual SDK2.10.1 calls,
eleven independent mathematical stages, all-PE raw audits and an eight-call
original vecmat control pass. Admission: `evidence/qualification-20260908T010554710275Z.json`
at the ports root.107corruptions are rejected;312unit tests pass.
The prior Q32 native failure is preserved. Q now uses block4 under unchanged
2%L2/3%peak/1%probability-mass gates; K/V use32.

See `docs/PROJECTED-CACHE-ATTENTION.md` at the ports root for mapping, resource
handoffs, bounds, source repairs, evidence scope and current limitations. This
is not complete Decode: no cache append, causal mask, automatic positions or
multihead/GQA mapping is implied.


All eight calls take65,606maximum-PE cycles, versus66,854for the matched source
local-compute control (ratio0.98133). Source zeroing/base initialization are inside
that interval. Static memory is46,592/source47,744bytes perPE. This is simulator
local-compute evidence, not complete Decode or real-wafer throughput.

The current native-only guard build is `run-20260908T003141050852Z`; its source,
schedule, actual native stdout and nine CSL files match the executed run. Failed
Q32, intermediate guard builds and all source/control evidence remain preserved.
