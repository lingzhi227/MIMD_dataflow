# Native SDK stencil baseline

This experiment invokes the **unchanged** SDK `stencil_3d_7pts` CSL library on
HLS-generated inputs. It establishes a native-library integration boundary; it
does not replace the production grid backend or mark additional applications
as ported.

## Reproduce

From the remote ports workspace, with SDK 2.10.1 installed:

```sh
.venv/bin/python experiments/native_stencil/baseline.py \
  projects/sdk_examples/resident_stencil_2x2x32_t4/run-20260906T082540504533Z \
  --block 8
```

The command creates a fresh `evidence/native-stencil-*` directory. Failed runs
are preserved. `--timeout` bounds compilation and execution; fatal simulator
messages stop the owned job. The worker runs inside the SDK container. The
outer audit compares every timestep with the checked HLS f32 interpreter.

```sh
.venv/bin/python experiments/native_stencil/baseline.py \
  evidence/native-stencil-TIMESTAMP --audit
```

The audit loads the frozen interpreter from the run and checks all snapshot
hashes before interpreting. `manifest.json` records upstream source hashes,
adapter snapshots and, in current runs, the SDK image/wrapper fingerprints.
`library-contract.json` records the checked integration contract. Earlier
exploratory runs predate these additional metadata fields; their original
snapshots are retained rather than rewritten.

## Adapter boundary

The SDK benchmark's `kernel.csl` and `layout.csl` are copied and adapted with
single-occurrence checked source anchors. Changes add a completion callback,
resident repetition, and exported timestep history. **The imported library
files are not patched.** Source adaptation is confined to this experiment;
it is not the proposed production lowering strategy.

The callback copies the completed output into the next resident input, records
history with memory DSDs, and invokes the next library call. The host supplies
each epoch's initial field and coefficients and reads the final field/history.
There is no host intervention between timesteps. The native library supplies
blocked neighbor communication, explicit DSR ownership and FMA instructions.

The official library configures local colors/routes and expects a contiguous
rectangle. It cannot safely be dropped into an arbitrary `SdkLayout` actor
region without reconciling those routing owners. Its coordinate convention is
also different: HLS array order is `[x,y,z]`; native host order is `[y,x,z]`,
and south/north coefficients are exchanged. Distinct directional coefficients
and random fields test this mapping.

`toolchain/library_contracts.py` checks the seven matching coefficient/plane
products structurally, the coefficient shape, memory and history-offset bounds,
and explicit permission for FMA/reassociation. It records the resource choices
of the original benchmark wrapper. This is **not yet a generic resource allocator**
or automatic production backend selection. A successful contract does not waive
compiler, simulator or numerical validation.

The current benchmark adapter requires **width > 1**: its transitive allreduce
module has a compile-time assertion even though this experiment does not call
the benchmark's synchronization entrypoint. The 1×1×7 attempt preserved that
compiler failure; the contract now rejects it before SDK invocation. This is a
benchmark-wrapper restriction, not evidence that the stencil library cannot
support a single PE. Removing unused benchmark dependencies belongs in a clean
production library adapter. The existing HLS backend still supports 1×1×7.

## Numerical and timing policy

HLS currently specifies center-first separate f32 multiply/add operations.
The SDK library uses a different order and FMA. Mathematical equivalence is
checked at every timestep with `rtol=3e-5`, `atol=3e-6`; bitwise identity is not
claimed. The contract refuses implicit relaxation of the numerical policy.

Both paths have resident timesteps and audit histories. Their PE placement,
fabric dimensions, I/O protocols and instrumentation transport differ. Total
simulator cycles are recorded separately and **must not be divided to claim a
controlled speedup**. A performance comparison still needs matched placement,
I/O, numerical policy and timing scope, or separately calibrated compute timing.
No hardware performance is measured here.

## Next production work

1. Represent native library calls, their layout ownership and asynchronous
   completion in the scheduled IR; do not dispatch by application name.
2. Introduce an explicit frontend/compiler floating-point policy before enabling
   this implementation as an optional lowering.
3. Compose library regions with external streams and collectives without
   overlapping colors, queues, microthreads or DSR lifetimes.
4. Preserve the scalar/ordered HLS route and all intermediate audit stages.

Upstream: Cerebras SDK examples commit
`4866cf330333446cb5e529e10f36be4600d1df29`, Apache-2.0. Copied source files retain
their original notices. The experiment does not imply Cerebras endorsement.
