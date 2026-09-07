# Explicit DSR leases for inference stages

The pinned `inference_comm.csl` uses separate memory and fabric DSR endpoints. Earlier schedule metadata listed memory-side3/4 but omitted fabric-side5/6. Generated CSL already reserved and used all of them; this correction changes reporting and composition checks, not executed instructions.

| Stage / endpoint | Explicit DSR banks and indices | Lifetime |
| --- | --- | --- |
| Local projection / gating | dest1, src0-1, src1-1 | Local compute; gating starts after projection join |
| Left memory | src1-3, dest3 | Overlapped exchange |
| Right memory | src1-4, dest4 | Overlapped exchange |
| Left fabric | dest5, src1-5 | Overlapped exchange, including preshift |
| Right fabric | dest6, src1-6 | Overlapped exchange |
| Score root reduction | dest1, src0-1, src1-1, dest2, src0-2 | Synchronous reduction after local score compute |
| Row collective vector | src1-2 | Synchronous row max/sum |
| Local normalization | dest/src0/src1 at1 and2 | Phase-exclusive with projection and row collective |

Here `src0-1` means source-zero bank, index1; `src1-5` means source-one bank, index5. Index alone does not identify a register. Vertical-only score exchange uses right memory4 and right fabric6; a full projection needs both axes. Compiler-managed temporary registers and SDK launch/memcpy resources are outside this explicit `@get_dsr` inventory.

`toolchain/inference_resources.py` supplies bank-specific leases. It checks simultaneous local projection and exchange leases for collisions. Tests compare the records against actual declarations and transfer operands in the pinned source, and prove that corrected MLP schedules emit identical files to the executed counter bundle. Existing frozen evidence is preserved with its historical metadata; use its own frozen debugger/auditor. A current planner may correctly reject equality with an older, incomplete resource plan.

`evidence/inference-dsr-metadata-20260907T050232183814Z.json` reconciles seven lowering families (projection, normalized projection/fanout, score, score-softmax, attention and MLP). For every representative, generated files are byte-identical and arithmetic/storage/schedule fields are unchanged. Older normalized plans additionally receive the previously introduced declarative rectangular projection-stage contract. This report is metadata reconciliation, not a new SDK execution or a claim that every register temporary has been inventoried.
