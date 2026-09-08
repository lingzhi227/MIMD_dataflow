# Feature-sharded batched RMS

Public `hls.cpp`: three batch rows, 512 features, 8×8 PEs, two grouped reduction
trees. Features are sharded along Y; every X column holds a replica. Local
vectors are batch-major. B=3 intentionally does not divide the mesh width.
One extra zero half lane pads the collective to four lanes; it is not a batch.

Standard bundle `run-20260907T160529124015Z` completed eight SDK2.10.1 calls.
Independent original x/w mathematics, all raw replicas, local/reduced sums and
padding pass. The pinned repaired Decode source matches five raw port groups
across eight calls. 28 injected corruptions are rejected. Linked static SRAM is
8,832 bytes/PE (not a runtime stack watermark). Max-PE simulator cycle overhead
against the matched source is 1.359–1.362%, including instrumentation and excluding
host I/O. Source repair and provenance remain explicit; this is not full Decode.

The same public C++ passes actual native checks on macOS and the SDK host.
`qualification-20260907T161741173353Z.json` under ports/evidence records admission.
The earlier160212100816SDK compile failure is preserved: `layout` was used as an
identifier; the corrected generated program uses `layout_mod`.

From the ports root:

```sh
python3 run_ports.py --select-exact waferllm/batched_rms_3x512_8x8_g2
# Configured SDK host:
HLS_CLANGXX=/usr/bin/clang++-17 .venv/bin/python run_ports.py \
  --select-exact waferllm/batched_rms_3x512_8x8_g2 \
  --sdk --sdk-threads 8 --sdk-suppress-trace --sdk-timeout 1200
python3 toolchain/debug.py projects/waferllm/batched_rms_3x512_8x8_g2/run-20260907T160529124015Z \
  --check-completed --node p7_7 --epoch 7 --step 1
```

`step`0/1/2 selects local sum/grouped sum/normalized data. Padding is labeled
separately. This first frozen batched auditor requires a complete saved batch
for validated inspection; ordinary raw inspection can show partial observations.

The runtime uses `batched_rms_local.csl` and a source-extracted generic
`axis_grouped_reduce.csl`, with five colors5–9 and queues3–7. Local DSR1/2 work
finishes before the collective reuses src1DSR2. No application task or microthread
is allocated. Y routes are installed once in init_task before any work, and
remain unchanged across warm calls. A local broadcast return is not a mesh-wide
join; in-call axis switching is deliberately not part of this profile. Future
fanout fusion must retain FIFO message order and explicit descriptor extents.
