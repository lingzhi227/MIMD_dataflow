# Numerical acceptance is separate from floating-point rounding correctness

The original `rtol=3e-5, atol=3e-6` accuracy screen remains unchanged for all
previous profiles. SUMMA additionally reports a named componentwise arithmetic
contract. Passing that contract **does not mean passing the fixed accuracy
screen**. Both results are retained in `audit.json`; `passed` for this profile
means the explicit arithmetic/dataflow acceptance contract, not a promise of
application-specific accuracy.

The preserved CPU run `run-20260906T092503821817Z` failed the fixed NumPy screen
on a 128×256 times 256×128 product, after native C++ and the source-order f32 IR
agreed. Maximum absolute difference from float64 was 1.3069549829936022e-5.
It remains a failed historical run. Inputs were not rescaled to make it pass.

## Componentwise contract

For a length-K dot product, standard rounding analysis relates absolute error
to the sum of absolute products, rather than only to the possibly cancelled
answer. The usual notation is gamma_n = n*u/(1-n*u). See the
[UT Austin ALAFF derivation](https://www.cs.utexas.edu/~flame/laff/alaff/chapter06-stability-dot-product-results.html)
and [Cornell numerical-analysis notes](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-02-04.html).

`toolchain/roundoff.py` conservatively takes n=2K+1 and u=2^-24 for f32
round-to-nearest. It computes a per-output budget from gamma_n*(abs(A)@abs(B)).
The n choice overcounts operations for FMA and is deliberately conservative;
this worst-case bound is not an estimate of likely error or an accuracy target.

The validation contract:

- Requires actual f32-quantized, finite A and B, matching shapes and n*u<1.
- Rejects an absolute-product sum exceeding finite f32 range, conservatively
  excluding intermediate overflow for the supported accumulation.
- Rejects subnormal inputs because input flush/DAZ behavior is not yet probed.
- Adds n*tiny_f32/(1-n*u) for possible intermediate underflow/flush loss.
- Includes a float64 reference accumulation allowance using u=2^-53.
- Uses two f32 error allowances when comparing two computed f32 paths, but
  separately compares **each path** against a float64 reference.
- Reports absolute error, error/budget and the unchanged fixed-accuracy result.

The budget and the magnitude calculation are validation tools, not generated
kernel arithmetic. Compiler/backend code does not use them to calculate C.
This policy is restricted to the explicitly relaxed SUMMA matmul profile; it
has not been substituted into solvers, factorizations, GEMV or other kernels.

## Structural and adversarial checks

A worst-case envelope can miss small dataflow faults. It is not the sole check.
SUMMA also verifies regenerated schedule/CSL hashes, native C++ against the
same-order IR, every PE's accumulated C tile after every prefix of K, and exact
agreement between the final tile histories and host output. Each prefix uses
its actual accumulated dot-product length, not the final K.

Four external calls exercise random inputs, coordinate-coded selection, zero
A after nonzero work (exact stale-state witness), and two positive halfway
addition cases distinguishing round-to-nearest-even from directed rounding.
The halfway and zero cases have exact output checks in addition to the budget.
Their execution is a target witness, not a proof of every floating-point edge
case; underflow/DAZ and overflow remain restricted as above.

Unit corruption checks reject a missing panel, wrong tile and stale C using
independently constructed expected products. Independent review may accumulate
the stored device inputs with `math.fsum` or exact arithmetic; none of these
checks should be presented as proving arbitrary-input numerical accuracy.
