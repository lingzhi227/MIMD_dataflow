# Isolated axis handover experiment

Prepared protocol experiment only. No application lowering imports this module.
The control ring is an O(P²) correctness/stress baseline, not a recommended
production barrier. SDK teardown-aware and hierarchical alternatives must be
compared before any performance-sensitive application admission.

A static Hamiltonian cycle uses alternating colors10/11, input queue2 and output
queue2. SDK memcpy uses queues0/1; the tested Decode grouped data plane uses3–7.
The separate control routes never change during data-axis handover.

1. Every PE finishes its old blocking data operation before participating in a
   ready round. Root receives the returning token only after every PE participated.
2. Root installs its new routes and starts a configuration token. Each other PE
   waits for that token **before** installing routes, then forwards it **after**
   installation. Forwarding ready alone never authorizes configuration.
3. Root starts release only after configuration returns. Each PE forwards release
   before starting new data work; root drains the returning release before reuse.

Tokens carry monotonic sequence numbers; begin/end pairing is checked. The proof
is conditional on participants representing every old-data destination and on the
CSL blocking move/queue behavior, which must be exercised in the actual SDK.
Local queue-empty observations alone are not a global-quiescence proof.

The planned experiment alternates grouped reductions Y→X→Y, changes DSD extents,
uses signed association-sensitive values, introduces per-PE skew before readiness
and before route installation, and repeats eight host calls. Preserve compilation
failures, all-PE outputs, sequence counters, queue masks and timing. Compare final
values with an independent axis-aware target model. Passing this bounded probe
would not qualify a full Decode application or establish scalable performance.

Executed follow-up: both ring g2/g4 probes passed eight calls, but independent
SDK X/Y collective planes are the preferred application integration candidate.
See `docs/DECODE-AXIS-HANDOVER-DESIGN.md` for exact runs and precision/performance
scope. `library_probe.py` checks sdk_axis_reduce odd-length/aliasing behavior;
`stable_silu_probe.py` exhaustively checks finite-half intrinsic behavior. These
are semantic probes, not additional qualified applications.
