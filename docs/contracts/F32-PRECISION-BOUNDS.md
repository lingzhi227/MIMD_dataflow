# Mixed precision range contracts (development)

These contracts bound storage range. They do not certify application accuracy,
ULP accuracy of arbitrary SDK functions, or hardware performance. The original
11-input mathematical oracle remains separate from the compiler interpreter.
The scoped mixed backend is connected to the compiler; catalog qualification remains separate.

## Native arithmetic

`float32_arithmetic.py` interprets the explicit public C++ operations. Full f32
contractions use fused multiply-add. A binary64 product of two binary32 inputs
is exact, but the subsequent binary64 addition can round onto a binary32
midpoint. Those lanes are resolved using platform `fmaf`; simply casting a
binary64 multiply/add is not an FMA model. Scalar, subnormal and double-rounding
witnesses are regression tested. Native `expf` and `sqrtf` are not substitutes
for the target SDK math implementation.

## Pinned SDK square root allowance

Reference: `references/sdk-math-2.10.1/math.csl`, SHA-256
`ef7233d3f2a43be553b512d0579ce284030f448a68b78b7b933814c48c2410c0`.
The function normalizes a positive finite input to [1,4), uses the magic-bit
inverse-square-root seed, performs two inverse-square-root refinements, then
one square-root correction and a power-of-two rescaling.

`evidence/sdk-sqrt-f32-seed-certificate-1401.json` enumerates all 16,777,216
normalized binary32 inputs. The observed seed ratio `r=y*sqrt(x)` is enclosed
by [0.963,1.037], with ample margin for the binary64 calculation of that ratio.
This is exhaustive for the seed interval only: it does not execute the CSL
function for all inputs and is not a full target square-root certificate.

`toolchain/precision_math_contracts.py` now checks the normalized forward-error
calculation with rational interval arithmetic. Each operation tracks an ideal
interval and propagated absolute rounding error; omitted roundings from FMA
fusion are covered by the unfused bound. A conservative absolute allowance also
covers flushed subnormal intermediates. Two inverse-square-root refinements
produce ratio error bounds 0.0020799375 and 0.0000074983. The final correction's
relative error bound is below 0.000002981, inside the reserved `2^-16` allowance.
The initial enumeration and this rational derivation establish different parts
of the argument; neither is an exhaustive execution of `sqrt_f32`.

The source-expression proof assumes binary32 round-to-nearest arithmetic,
truncating float-to-i16 conversion and exact normal power-of-two scaling.
Actual SDK probes additionally bind observations to SDK 2.10.1's pinned SIF:
`evidence/f32-math-20260907T143628432833Z/review.json` checks 4,096 square roots
across mantissas/exponents, 4,096 exponentials in [-12,0], and all range-reduction
casts. Largest sampled relative errors are 8.211e-8 for sqrt and 1.530e-6 for exp.
These are sampled target observations, not universal ULP guarantees.

## Exponential range versus accuracy

The rational source-expression derivation bounds `exp_f32` as positive, finite,
normal and below 3 on [-80,0], and establishes the exact zero image 1. It bounds
argument reduction, the Horner polynomial, table multiplication and exponent
scaling. These properties suffice for a finite nonnegative softmax range; they
do not establish an application accuracy tolerance over that whole domain.

The first actual probe over [-80,0] failed its predeclared 2e-6 relative-error
threshold: error 4.549e-6 at -79.70696. That failed gate is preserved in
`evidence/f32-math-wide-domain-failure-continued.json` and its original SDK run
`evidence/f32-math-20260907T142916015298Z`. No threshold was relaxed.

The current kernel's derived shifted-logit magnitude is at most 10.000001908.
The scoped backend rejects a derived domain exceeding [-12,0]. A separate fresh
SDK probe in [-12,0] passes the same 2e-6 threshold; the complete application
also retains its original eight independent branch/final accuracy cases.
Do not infer a global exp accuracy contract from either the 519 in-domain
samples in the first probe or the second 4,096-point probe.

## RMS correlation and narrowing

Bounding a normalized output as `max_input/sqrt(epsilon)` loses the correlation
between the input and its squared norm. Instead retain the nonnegative square
sum through local reduction and the row collective. `half_output` uses a
conservative reduction depth, a positive epsilon floor, a lower rounding factor
for the squared norm, and the SDK square-root allowance above. Gamma is applied
before the inverse in f32, then the result narrows to half. The returned bound
rounds outward and refuses a potentially overflowing half result.

The finite square/sum range and precision of every communication path must be
checked separately. This helper cannot qualify a profile by itself. In
particular, a range derived for the old half arithmetic must not be silently
reused as proof for the new f32 graph, even when its numeric bound is larger.
