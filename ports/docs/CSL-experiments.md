# CSL constraints established by SDK experiments

Results below are specific to the pinned SDK 2.10.1 image and WSE3 target. Compiler acceptance and runtime protocol correctness are separate evidence levels. Named historical run paths refer to the external experiment archive when not bundled.

## Calling the native stencil library

The [native baseline](../experiments/native_stencil/README.md) calls unchanged SDK stencil library files and checks 64 PEs, 16 resident steps, two inputs and 262,144 intermediate values. A block=7, z=32 case tests tail blocks and DSR length reset on the next call. FMA/order differences are checked with explicit tolerances, not bitwise identity.

The library configures its own local colors, so it cannot be arbitrarily embedded in another layout region. SDK north/south coordinates differ from the HLS array convention; coordinate and coefficient conversion is part of the ABI. A 1×1 attempt failed in a transitive benchmark allreduce assertion requiring width>1. This is a wrapper restriction, not evidence that the stencil algorithm cannot run on one PE.

## Compiler resource probes

Matched controls change only the resource under test.

| Resource | Accepted index | Rejected index | Scope |
| --- | --- | --- | --- |
| Color | 23 | 24 | Matched program |
| Input queue | 7 | 8 | Matched program |
| Output queue | 7 | 8 | Matched program |
| Microthread | 7 | 8 | Explicit UT in a move |
| Local task | 30 | 31, 32, 7 | Compiler requires [8,31) |

These are index legality boundaries, not permission to allocate all resources simultaneously. Routes through intermediate PEs consume resources too.

## Compiles but fails: microthread ownership

After switching to full-frame asynchronous receives, a run failed with `trying to term ut_instr[4], but it's not ours`. The corrected contract assigns neighbor sends to UT0, initial forwarding to UT1, final gathering to UT2 and neighbor inputs to UT3–6. Default allocation is not a safe general policy.

Local send completion releases local buffer ownership; complete reception establishes halo readiness. Both are required before advancing. Validation checks every timestep and accumulated traffic, not only the final output.

## I/O, scale and diagnostics

For 16 PEs, column distribution/gathering reduced external I/O tiles from 32 to 8 and fabric from 14×32 to 14×14. Distribution, gathering and intermediate updates execute in CSL. The old 64-PE attempt timed out at 240 seconds; a smaller I/O layout and larger explicit budget later passed. Timeout, compile failure, simulator fatal and numerical mismatch are recorded separately.

The runner detects simulator fatal messages and terminates its owned process group to avoid waiting indefinitely on receives. It preserves stage/reason records. Raw traces reached roughly 16GB in one experiment; bulk traces and full run bundles are excluded from this curated release.

No hardware throughput result, universal route-safety proof or complete matched hand-written-kernel performance comparison follows from these probes. Multiple fields, additional boundaries/radii and arbitrary region composition remain future work.
