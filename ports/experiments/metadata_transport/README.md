# WSE3 mixed-width metadata transport probe

This independent two-PE experiment isolates `u16 header → u16 rows → f32 tail`.
It is a target-specific regression probe, not an SpMV performance benchmark.
The receiver checks every row marker and all three f32 sentinels.

The SDK image used is 2.10.1, SHA256
`fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d`.
In `metadata-probe-20260906T122546645806Z`, sending 26 u16 markers from an
odd offset inserts a zero after marker 1007 at the receiver. Its final marker
is then consumed as an f32 word. This reproduces the actual SpMV trace mismatch.

`metadata-matrix-20260906T122854645516Z` shows offset-zero length26 passes;
offset-one length26 fails with or without receiver delay and with matching
output DSD extent. Length7 passes, while8/9/25/27 fail in the tested odd-offset
configuration. This does not establish a general hardware erratum or a magic
packet-length rule. Authoring failures (CLI bool handling, launch path, and
comptime pointer casts) remain separately preserved in adjacent run folders.

`metadata-matrix-20260906T123258211899Z` passes all24 combinations:
lengths7/8/9/25/26/27, (offset1, delay2000) or (offset3, delay0), and each of:

- Explicit pairs of u16 packed into u32, received as u32 then unpacked.
- Word-aligned staging followed by the original u16 send and receive format.

Both use the actual shared `runtime/u16_transport.csl` staging implementation.
The second was selected for SpMV to preserve its logical wire format. Each
logical count remains unchanged; padded backing words are never counted as
additional row elements. The local send callback owns scratch reuse.

Run `matrix.py` with the outer Python environment on workstation. It copies
sources into fresh evidence directories before invoking the SDK. `exact_length`
is the historical numeric probe selector:0 original,1 matching fabric extent,
2 explicit u32 wire,3 aligned u16 wire. Production HLS does not expose this
experimental selector. SDK options, sources and outcomes are retained per case.
The production 16-PE and64-PE four-epoch tests additionally exercise repeated
staging reuse with changed sparse structures. ELF/LMA mapping is checked by
`../audit_spmv_alignment.py`; an ELF may represent several PEs.
