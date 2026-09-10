# From a numerical program to a bounded PE protocol

This is the accompanying prototype for the [spatial HLS survey](https://github.com/lingzhi227/dataflow-programming-literatures). It extends the existing SUMMA implementation with a finite event-protocol IR and an exhaustive checker. It does not add a new GEMM algorithm or claim an automatic proof of CSL.

The survey's proposed compiler covers constructive algorithm selection, localization, schedules and target interfaces. This prototype exercises one bounded execution boundary of that larger design. The [frozen evidence folder](evidence/2026-09-10/) contains the actual inputs, outputs, generated CSL and audit from the completed experiment.

Run from the repository root:

```sh
python3 tools/check_spatial_contracts.py --output build/research/spatial-contracts.json
python3 tools/check_spiral_formula.py --output build/research/spiral-formula.json
python3 -m unittest discover -s tests/unit -p test_event_protocol.py -v
```

The command obtains the existing four-round SUMMA plan, constructs the protocol, checks every reachable abstract interleaving, and freezes the exact protocol inputs and source hashes in its report. Failed variants are intentional mutations of this abstraction; they are not claims of defects in the existing CSL template.

| Check | Result | Meaning |
|---|---|---|
| Four-round paired-completion SUMMA | PASS; 131 states, 202 transitions | Every maximal execution of this finite abstraction terminates without a modeled ownership/version error |
| Remove the wait for B | UNSAFE | A shortest witness reads B before its required publication |
| Start the next panel without the previous compute completion | UNSAFE | A witness observes an overwritten version or overlaps an active access |
| Two ordered streams, A capacity 1 | DEADLOCK | Producer waits for A space while consumer waits for B data |
| Same streams, A capacity 2 | PASS | Every modeled execution completes |
| Unfolded systolic schedule | PASS; 192 dimension/latency profiles | Exact integer results and explicit dependence distances agree |

A state limit produces **INCONCLUSIVE**, never PASS. The generated report preserves shortest failure/deadlock traces. Unit checks cover fanout lifetime, mismatched completion ownership, leaked ownership, and state-limit handling as well as the paired repairs.

The separate four-point FFT checker uses exact Gaussian-integer coefficients and symbolically executes a fixed two-PE layout. Removing or misrouting an exchange fails. It does not run SPIRAL, generate FFT CSL or establish physical routing feasibility.

The completed SDK run reuses the existing 64×64×64 SUMMA profile on 4×4 PEs for four batches. All 16,384 final outputs and 65,536 intermediate observations match the frozen reference for these small integer-valued inputs. Native compilation and byte-for-byte CSL regeneration passed. The measured 130.35 seconds include compilation and initialization; they are not device latency. See the [numerical audit](evidence/2026-09-10/audit.json) and [measurement scope](evidence/2026-09-10/README.md). This is one new bounded run, not a rerun of the full historical catalogue.

## Mathematical program and two realizations

For M by K and K by N matrices, define C(i,j,0)=0 and

`C(i,j,k+1) = C(i,j,k) + A(i,k) B(k,j)` for `0<=i<M, 0<=j<N, 0<=k<K`.

In exact arithmetic, induction on k gives C(i,j,K)=sum_k A(i,k)B(k,j). The executable mapping check uses small integers whose products and sums also fit int32; it does not replace floating-point validation.

An unfolded systolic realization places instance (i,j,k) at PE (i,j), with logical issue time `theta=i+j+L*k`. Assume independent operand forwarding, one logical time unit per mesh hop, and a MAC recurrence latency of L. Inject A(i,k) on the west boundary at i+L*k, and B(k,j) on the north boundary at j+L*k. Both reach (i,j) at theta. The accumulation dependence has distance L, and the two forwarding dependences have distance one. Every PE starts its successive MACs L units apart. These are logical scheduling assumptions, **not measured CSL cycle counts**.

Mapping all instances of a 2 by 2 grid to one single-issue PE while keeping theta is invalid: (0,1,0) and (1,0,0) both require that PE at time one. A legal dependence order alone is insufficient; folding also needs serialization, storage and a feasible communication schedule. Link contention, instruction issue, DSR availability and local bank conflicts are outside this simple check.

The implemented backend is **blocked SUMMA**, a different realization of the same recurrence. For a P by P grid and M=P*Mt, K=P*Kt, N=P*Nt, PE (x,y) stores C[y*Mt:(y+1)*Mt, x*Nt:(x+1)*Nt]. In round r it receives the A panel from column r of its row and the B panel from row r of its column. It accumulates Kt terms in increasing global k=r*Kt+k_local order. A proof by round induction gives the partial product over k<(r+1)*Kt. No unit-hop systolic timing claim is transferred to this collective implementation.

## Operational model and proof boundary

The protocol IR contains finite actor programs and named one-shot events. A buffer has a published version (initially -1), at most one in-flight writer, and a set of active reader tokens. `write_begin` requires no writer or reader; `write_end` requires the same actor, token and version; `borrow` requires the requested published version and no writer; `release` relinquishes exactly that actor's borrow. A `wait` is enabled only after the corresponding `signal`. Concurrent readers can coexist; every reader must release before the next write. These checks describe access ownership, not arithmetic.

For each round, actors X and Y acquire and fill A and B independently, then signal their respective completion events. Compute waits for both events, borrows both panels, computes, releases them, and signals the end of the round. The next round waits for this event before overwriting either panel. The protocol abstracts a whole panel transfer into begin/end events and the synchronous local computation into the interval between borrows and releases.

**Conditional argument.** In round zero, X and Y need no prior event. Each has finite work with a completion step. Once both signals occur, compute can acquire two correctly versioned panels, and no later writer is enabled before its round-completion signal. Compute therefore releases both panels and enables the next round. Induction establishes progress and safe reuse for any finite number of rounds under eventual completion of each collective, execution of enabled tasks, and absence of external reset/failure. The model checks all finite actor interleavings for the selected P=4 instance. It does not establish the assumptions about SDK internals.

The state explorer adds no idle transition: each modeled step increases one actor's program counter. Thus the sum of remaining instructions strictly decreases. In a finite protocol, absence of reachable error or unfinished terminal state implies safe termination of every maximal modeled execution. This conclusion is stronger than finding one successful trace, but narrower than proving an arbitrary runtime live. If the real environment can indefinitely postpone a completion, the model provides no unconditional progress guarantee. There is no fairness-based rescue of an already deadlocked state.

The abstraction assumes that a collective completion means this PE's panel is ready and the library no longer accesses that storage. This boundary must be checked against the selected library and target. It does not equate a primitive local send completion with remote application consumption. The existing scalar communication library explicitly makes the same distinction.

## Exact counterexample: acyclic data graph, bounded deadlock

Use two FIFO edges a and b from producer P to consumer C, both initially empty. P executes `put(a); put(a); put(b)`. C executes `get(b); get(a); get(a)`. P produces exactly the counts C consumes. The data graph is a DAG; its balance equations hold for one firing of each coarse actor.

With capacity(a)=capacity(b)=1, the only enabled first operation is P's first put(a). Now P cannot execute its second put(a), while C cannot get(b). Both are blocked. The shortest witness consists of the single successful put; the report identifies the unfinished state. Capacity(a)=2 repairs this particular schedule without changing its token order. Alternatively, changing P to `put(a); put(b); put(a)` permits capacity one. Those are different implementation choices, and neither follows merely from rate balance.

FIFO `put/get` are atomic publication/reclamation in this counterexample. They are deliberately separate from the buffer borrow model. A multi-step DMA implementation needs explicit reservation and completion events and may retain an input beyond its logical consumption.

The checker represents FIFO occupancy, not token values, and does not require all channels to be empty at general protocol termination. The selected balanced example ends empty. Request-drain or output-consumption requirements need explicit additional acceptance conditions; a generic protocol PASS does not establish them.

## Lowering to the existing CSL program

The frozen publication baseline is `dc14c0df54b12cb9e47ae710e2215084154c9570`.

- `lib/Conversion/mesh_gemm.py:plan` sets the tile sizes, increasing-K numerical policy, per-PE memory estimate and the paired completion dependency.
- `runtime/csl/mesh_gemm_pe.csl:x_done` activates the initially blocked compute task; `y_done` unblocks it. Either order leaves compute executable only after both callbacks. `compute` blocks itself before starting the next round. The finite IR's one-shot events abstract these task bits; they are not a translation of general CSL task semantics.
- X uses queues 2/4 and DSR index 1 in each bank; Y uses queues 3/5 and DSR index 2; memcpy uses queues 0/1. They are disjoint by construction in this profile. This restriction is conservative, not an assertion that all hardware queue sharing is impossible.
- `runtime/csl/mesh_gemm_vector.csl` uses memory DSDs and `@fmacs`, retaining the local reduction order. The backend is based on the Cerebras SDK example, whose copyright/license remain in the files.
- `lib/Runtime/mesh_gemm_sdk.py:audit` regenerates the CSL byte-for-byte and checks every saved round, as well as the final matrix, against an explicit floating-point error envelope.

The newly added protocol is constructed from the plan and a manually reviewed template relation. It is not automatically extracted from generated CSL. A future compiler pass must maintain this relation through transformations or validate it after lowering; the current checker alone cannot detect arbitrary manual edits to a CSL file.

## Numerical relation

Exact equality over integers or complex algebra does not license floating-point reassociation. For binary32 round-to-nearest, take a=2^24, b=1 and c=-2^24. `(a+b)+c` yields zero; `a+(b+c)` yields one. The checker reproduces both outcomes with explicit binary32 rounding. The source's relaxed FMA contract and the existing componentwise dot-product envelope therefore remain necessary. A protocol PASS does not imply an error tolerance has been met.

## Related formal models

Kahn's stream semantics and Lee/Parks' dataflow process networks describe functional determinacy under their channel and process assumptions. Replacing unbounded nonblocking writes with finite blocking writes adds a resource constraint. For SDF, a positive repetition vector satisfying rate balance is necessary for a bounded periodic execution; it is not sufficient for liveness. Stuijk, Geilen and Basten's 2008 timed CSDF model explicitly represents storage by reverse credit channels: output space is reserved at firing start, while input space returns at firing end. Its chosen timing abstraction is conservative under stated earlier-release/later-reservation conditions. Our begin/end and borrow/release distinction follows this existing line of reasoning; it is not a new discovery.

Primary sources: [Lee and Parks, 1995](https://ptolemy.berkeley.edu/publications/papers/95/processNets/), [Stuijk et al., 2008](https://sstuijk.estue.nl/tools/sdf3/publications/tc_csdf_buffersizing.html), [CSL task IDs](https://sdk.cerebras.ai/csl/language/task-ids), [CSL DSD reference](https://sdk.cerebras.ai/csl/language/dsds), [WSE-3 microthreads](https://sdk.cerebras.ai/csl/language/microthreads_wse3). Retrieved 2026-09-10. The DSD and microthread pages contain conflicting guidance on concurrent queue sharing; this experiment does not try to resolve it by inference.
