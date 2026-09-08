# Decode-layout supplied pair transform

B5/N1024 on8×8 PEs, feature X partitions with Y replicas, batch-major storage and explicit source `odd_even` order. The frontend invokes shared `rotate_pairs` with broadcast supplied sine/cosine. It lowers to a reusable packed-half DSR CSL library, with complete product witnesses and repeated-call audit.

See [contract and evidence](../../../docs/BATCHED-PAIR-ROTATION.md). Qualified as bounded configuration140: six HLS/repaired-source SDK calls, two native hosts,491,520 product observations and30 rejected faults. HLS1,878/source2,081 maximum PE cycles; the source has an explicit odd-offset repair and timed immutable-input copy. The unrepaired failure and SDK DSD/alias probes are preserved. No position generation, head selection, cache append, complete Decode or hardware throughput claim.
