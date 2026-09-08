"""Shared shape/layout contract for CSL forward-aligned rectangular projections."""

from frontend import check


def stage(left_shape, right_shape, policy):
    check(len(left_shape) == len(right_shape) == 2, "projection matrix ranks")
    m, k = left_shape
    kk, n = right_shape
    p = policy["rows"]
    accumulation = policy.get("accumulation")
    check(accumulation in (None, "block_f32"), "projection accumulation policy")
    core = {k: v for k, v in policy.items() if k != "accumulation"}
    check(
        all(type(v) is int and 1 <= v <= 512 for v in (m, k, n)) and kk == k,
        "projection matrix extents",
    )
    check(
        type(p) is int
        and p in (4, 8)
        and core
        == dict(
            rows=p,
            cols=p,
            exchange="two_hop",
            initial_align="forward",
            reduce="local",
            overlap="double_buffer",
            fp="relaxed",
            compute="dsr",
        ),
        "forward projection shared CSL policy",
    )
    check(all(v % p == 0 for v in (m, k, n)), "projection divisible tiles")
    mt, kt, nt = m // p, k // p, n // p
    check(mt * kt % 4 == kt * nt % 4 == 0, "projection four-half communication packing")
    check(
        max(mt * kt, kt * nt, p * mt * nt) <= 32767,
        "projection signed DSD and observation offsets",
    )
    result = dict(
        M=m,
        K=k,
        N=n,
        P=p,
        Mt=mt,
        Kt=kt,
        Nt=nt,
        left_length=mt * kt,
        right_length=kt * nt,
        output_length=mt * nt,
        rounds=p,
        weights="static logical weights host-packed in row-major tiles with two-hop block-row alignment",
        activation="logical column-major tiles aligned forward on device",
        result="logical column-major tiles",
        right_descriptor="contiguous feature vector; contraction increment Nt",
        k_block_rule="cycle[(position(y)+position(x)-round)%P]",
        completion="both-axis transfer and local DSR compute joined before buffer swap",
    )

    if accumulation is not None:
        result.update(
            accumulation=accumulation,
            partial_type="f16",
            merge_type="f32",
            block_size=kt,
            output_rounding="f16 after final f32 merge; optional rounded prefix observations",
        )
    return result
