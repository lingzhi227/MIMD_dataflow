# SDK half numerical prerequisite probe

Preparation for distributed inference stages, not an HLS application port or
full RMSNorm/softmax qualification. The generated PE calls actual SDK exp_f16
and sqrt_f16 and a verbatim fast_exp function extracted from the pinned Decode
source. Values include signed zero, subnormals and negative/positive scores.
Changed/reversed/zero inputs test repeated host writes and completion.

The host must retain raw output bits and report the source polynomial versus
standard exponential discrepancy separately. It must not silently assign the
source polynomial the semantics of exp. Distributed reduction order, RMSNorm
buffer/index concerns and full inference require separate execution probes.

The driver prepares a fresh frozen directory locally. Execution is a separate
explicit step on the SDK host; preparation never marks SDK validation passed.
