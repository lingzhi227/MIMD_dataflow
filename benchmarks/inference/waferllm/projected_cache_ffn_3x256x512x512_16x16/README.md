# Resident projected cache attention + FFN

B=3, N=256, cache S=512, FFN F=512, 16×16 resident dataflow.
The source preserves the qualified attention equations, source-shared gamma,
new K/V auxiliary outputs (no append), and FFN residual Z+delta. The second RMS
explicitly uses a mean statistic; each contraction declares half blocks/f32 merge.

The 35-node source passes the normal Clang frontend, typed verification,
planning, native execution and CSL generation pipeline under
`mesh_projected_cache_ffn.v1`. The generated CSL reuses SDK collectives and half
math with a caller-owned attention region and FFN continuation. Both regions
share the SDK communication buffers and finish through one outer host callback.
The 23-phase plan tracks 83 logical value lifetimes and shared DSR/task ownership.

## Validation status

The standard frozen bundle is `run-20260908T030001167477Z`. Its full eight-call
SDK 2.10.1 run and the nine-contraction source-compute control are in progress;
this application is **not yet admitted** to the qualified catalog. All eight
native cases and 18 observed mathematical stages passed on both Apple Clang
and remote Clang 17, with exact public native output agreement. This does not
replace the device result checks.

The earlier bounded SDK experiment completed one call with all 49 non-timing
physical port groups matching the source-compute control. It was deliberately
stopped after observing that the original one-hour timeout was insufficient for
eight full instrumented readbacks. Both partial runs remain failed evidence;
the fresh full runs use a six-hour budget. No full-batch performance claim is
made from that first call.

## Numerical and resource contract

Q uses half block1/f32 local merge, K/V block16, cache-score block16, cache-value
block32, output projection block16, and UP/GATE/DOWN block4. Q block4 failed the
unchanged 3% peak score gate for a cancellation case; block1 passes both the 2%
L2 and 3% peak gates. The inputs and thresholds were not weakened.

The second RMS consumes actual resident Z and shared gamma. Its typed mean
statistic prescales local square sums in f32 before SDK reduction and half
narrowing. The last original-input case produces Z=33: a half global square sum
would overflow, whereas the mean remains representable. Dedicated eight-call
16-PE SDK probes validate the mean helper and its transition between SUM phases.

The current parent-derived range bound for Z is 34.0625. The planner reserves
44,030 bytes per PE including numerical storage, protocol and code/stack budget.
The first linked CSL probe measured 42,048 static bytes; it exposed the previous
39,934-byte estimate as insufficient. Static linked memory and a reserved stack
budget are different measurements; full-run ELF checks are still required.

## Source and scope

`SOURCE-MAP.json` binds all 18 mathematical stages to the pinned migrated Decode
source and records intentional arithmetic repairs. Old K/V cache is read-only;
rotated new K and projected new V are separate outputs. There is no cache append,
head/GQA, mask or automatic position semantics. The source control replaces nine
local contractions with pinned source bodies. RMS, pair transforms, softmax,
SiLU and communication are shared, so it is not independent full Decode timing.

See `../../../docs/MEAN-STATISTIC-RMS.md` and
`../../../evidence/composed-native-host-20260908T031354352108Z/report.json`.
