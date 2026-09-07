# Next numerical composition: rectangular gated MLP

Historical source-analysis plan. The implementation now exists and is undergoing SDK validation; see `RECTANGULAR-MLP.md`. It is not yet a qualified HLS profile. It follows the supplied-Q/K/V attention qualification in the numerical queue.

Pinned `Prefill/src/prefill.csl` distinguishes the attention output projection `h1_matmul` from MLP `z1_matmul`, `z2_matmul`, `z3_comp`, and `h2_matmul`. The MLP expansion uses Mt×Kt times Kt×Ft tiles; contraction reverses feature/FFN roles to Mt×Ft times Ft×Nt. Existing square normalized QKV fan-out does not establish this rectangular schedule.

The intended numerical graph is supplied normalized X and static weights U,G,D:

```text
up   = X U
 gate = X G
 hidden = up * SiLU(gate)
 output = hidden D
```

It should use existing C++ matmul, SiLU and multiply semantics. Static model weights may be packed into the source-prescribed ownership before execution, as with existing projection kernels. Dynamic activation stages must remain resident. Source `z2_matmul` resets left-buffer roles after the first projection while omitting a second preshift; the previous QKV experiment suggests a live-buffer ownership risk, but this must be independently executed for the rectangular path before claiming the same repair works.

Source `z3_comp` already uses CSL `@map` SiLU plus a DSR multiply. The previously qualified gated-activation domain and finite-half exp behavior must constrain the composed gate output, not just external X/weight bounds. The source-derived SiLU approximation is not uniformly accurate on all finite half inputs. Expansion/contraction intermediates and their independent mathematical checks require an explicit numerical contract.

The compiler should share a reusable rectangular projection stage contract with existing projection lowerings, preserving dimension-specific communication lengths, initial alignment and completion joins. Sequential up/gate branches may reuse left alignment only if live ownership is maintained. Hidden-to-down requires a fresh left alignment. Right DSD entry state must be explicit; the attention stride-leak failure is a reminder that changing length alone does not reset a descriptor.

Acceptance requires original and explicitly adapted source probes, native C++ plus independent original-input arithmetic, exact target-half expansion/gate/product/down prefixes, warm changed weights/inputs, bounded resources, all-stage PE memory accounting, corruption rejection and matched lean source controls. Full RMS/residual integration, complete prefill/decode and masking/head/cache semantics remain additional work. No stencil or physics expansion precedes this numerical queue.
