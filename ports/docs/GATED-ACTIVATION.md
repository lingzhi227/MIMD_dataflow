# Gated SiLU on CSL

The frontend describes two typed operations: `silu(gate)` followed by
`multiply(up, activated)`. Both carry the policy
`rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed`.
The first qualified shape under review is64×256; larger and odd-tile profiles
are separate execution gates, not implied by compilation.

## Source and lowering

The reference is Apache-2.0 WaferLLM Prefill `silu_kernel`/`z3_comp`, commit
`fd1c2daae37cd68706c03fc8009887ecee9900f8`, from
https://github.com/MeshInfra/WaferLLM. The isolated source control preserves
both function bodies and replaces the outer Prefill continuation with an empty
callback. It does not execute a full MLP, Prefill or Decode. Decode's quartic
`fast_exp` is a different algorithm and is not silently substituted.

`mesh_swiglu.v1` recognizes typed dependency edges, including independent input
declaration order and commutative product operands. It rejects aliased inputs,
unsupported edges/policies and invalid resource bounds. This is a bounded
subgraph lowering, not a general arbitrary graph fusion implementation.

The CSL backend uses SDK `<math>.exp_f16`, synchronous `@map` and DSR half
multiply. Each PE owns a column-major tile. There is no application fabric
exchange, user color, queue, local task or microthread allocation in this
pointwise phase. SDK memcpy owns launch/I/O resources. Explicit compute DSR
slot1 is loaded only after the synchronous map completes.

Host `up`/`gate` allocations retain their identity and contents across a call.
The private activation buffer becomes the product destination at its last use.
Sampled mode copies activation before reuse; counter mode omits that copy.
Source control instead transforms its gate input in place and keeps its product
separate. Their numerical schedule is the same, but buffer ownership differs.

## Numerical contract

Inputs are finite binary16 values of magnitude≤8. Native C++ uses stable
standard SiLU, stores the activation in half, then multiplies in half. Device
validation separately checks exact SDK-expression half bits and standard
mathematical accuracy. For each component, with `r = up * standard_silu(gate)`,

```
abs(device - r) <= 0.004 * abs(r) + 2^-24 * (1 + abs(up))
```

The absolute term accounts for stored half activation and final product
rounding, including subnormals. This is not a uniform relative-error or f32
accuracy promise. In the dedicated tiny-value epoch, relative L2 error can be
about0.378 while maximum absolute error is2.3842e-7. Reporting keeps the
absolute term visible instead of applying the generic f32 tolerances.

SDK probes cover all31744 nonnegative finite half magnitudes and the previous
negative-exp domain. The production signed exp/SiLU model matches253952
observed half words across those probes. Source `x/(1+exp(-x))` loses906
representable negative-tail results at magnitudes11.09375–20.328125 because
positive half exp overflows. An alternative half expression still loses192
cases. The first application bound≤8 is deliberate; full-domain standard
SiLU accuracy is not qualified.

## Evidence and debugger

`swiglu_64x256_8x8/run-20260907T000304825939Z` passed six SDK2.10.1 WSE3
simulator calls in one runtime instance with changing inputs. Audits check
immutable input bits, sampled activation bits, product bits, phase/epoch
completion and bounded48-bit timestamps. Eleven intentional corruptions are
rejected. Static linked high water is7040B, not a runtime-stack measurement.
The sampled original-source control has exact outputs and maximum local
interval28048/28026 cycles (~0.0785% HLS overhead). It includes the HLS
activation observation copy, which the source control obtains by aliasing.

The debugger exposes activation and final-product stages, mapped tile
ownership and measured/expected values. Counter activation is marked unobserved.
Each fresh bundle preserves frontend IR, checked/optimized IR, schedule,
runtime source, native stdout, implementation snapshot and actual SDK artifacts.
No real-hardware throughput or full-inference claim follows from these local
simulator intervals.

Counter64×256 bundle `20260907T001707252972Z` also passes six calls and11
corruption mutations. Matched source maximum-local intervals28011/28026 and
26751/26766 are effectively at parity; this small difference is not a broad
speedup claim. Sampled64×256 is registered by `qualification-20260907T002338013039Z.json`.

`gated-activation-domain-bound-review.json` uses the exhaustive observed SDK
SiLU outputs for all36866 signed half gate encodings of magnitude≤8. It checks
a sufficient error inequality over every real up magnitude≤8, conditional on
the usual round-to-nearest binary16 product bound. This is stronger coverage
than the six application fixtures, but is not exhaustive device enumeration
of every up/gate pair or a formal proof using exact real exp.

## Large-profile qualification update

128×1024/8×8 now completes six SDK calls and11mutation rejections. Registration: `qualification-20260907T005710490287Z.json`; sampled source control224048/223802 local cycles with identical bits. This is a bounded gated stage, not full MLP.
