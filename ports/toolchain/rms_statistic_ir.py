"""Typed boundary between local RMS statistics, SDK reduction and normalization.

This is a reusable lowering contract. A parent graph supplies a derived input
range and owns placement/leases. It is not a kernel-name dispatch or a backend.
"""

from dataclasses import asdict, dataclass
import math
from frontend import check
from mean_statistic_bounds import plan_mean_statistic, normalized_mean_bounds


@dataclass(frozen=True)
class StatisticType:
    representation: str
    feature_count: int
    divisor: int
    local_dtype: str = "f16"
    collective_dtype: str = "f32"
    result_dtype: str = "f16"

    def verify(self):
        check(self.representation in ("sum", "mean"), "RMS statistic representation")
        check(
            type(self.feature_count) is int and 1 <= self.feature_count <= 2048,
            "RMS statistic feature count",
        )
        check(
            type(self.divisor) is int
            and self.divisor
            == (self.feature_count if self.representation == "mean" else 1),
            "RMS statistic scale matches representation",
        )
        check(
            (self.local_dtype, self.collective_dtype, self.result_dtype)
            == ("f16", "f32", "f16"),
            "RMS statistic precision contract",
        )
        return self


def verify_connection(produced, consumed):
    produced.verify()
    consumed.verify()
    check(produced == consumed, "RMS statistic producer/consumer mismatch")


def mean_boundary(node, producer, gamma, *, derived_input_bound):
    """Compute, rather than accept, the range certificate of an internal edge."""
    check(
        node.get("op") == "rmsnorm"
        and len(node.get("inputs", [])) == 2
        and node["inputs"][0] == producer.get("id"),
        "RMS boundary actual producer edge",
    )
    check(producer.get("dtype") == node.get("dtype") == "f16", "RMS boundary dtype")
    shape = producer.get("shape")
    check(
        isinstance(shape, list) and len(shape) == 2 and node.get("shape") == shape,
        "RMS boundary shape",
    )
    b, n = shape
    check(
        type(b) is int and type(n) is int and 1 <= b <= 16 and 1 <= n <= 2048,
        "RMS boundary bounded shape",
    )
    policy = node.get("dataflow", {})
    p = policy.get("rows")
    check(
        type(p) is int and p in (4, 8, 16) and policy.get("cols") == p and n % p == 0,
        "RMS boundary region geometry",
    )
    expected = dict(
        rows=p,
        cols=p,
        partition="features",
        axis="y",
        layout="batch_major",
        reduce="sdk_axis",
        result="replicated_columns",
        accumulation="f16",
        collective="f32",
        statistic="mean",
        math="sdk_half",
        compute="dsr",
        fp="relaxed",
    )
    check(policy == expected, "RMS explicit mean statistic policy")
    check(
        type(derived_input_bound) in (int, float)
        and math.isfinite(derived_input_bound),
        "RMS parent-derived finite range",
    )
    statistic = StatisticType("mean", n, n).verify()
    check(node["inputs"] == [producer["id"], gamma.get("id")], "RMS actual gamma edge")
    check(
        gamma.get("op") == "input"
        and gamma.get("dtype") == "f16"
        and gamma.get("shape") == [1, n],
        "RMS gamma input type",
    )
    gamma_bound = gamma.get("abs_bound")
    check(
        type(gamma_bound) in (int, float)
        and math.isfinite(gamma_bound)
        and gamma_bound >= 0,
        "RMS gamma explicit input bound",
    )
    capacity = (b + 1) // 2 * 2
    result = plan_mean_statistic(derived_input_bound, n // p, p, n, capacity)
    normalized = normalized_mean_bounds(
        derived_input_bound, gamma_bound, n // p, p, node["epsilon"]
    )
    return dict(
        producer=producer["id"],
        consumer=node["id"],
        statistic_type=asdict(statistic),
        geometry=dict(batches=b, features=n, participants=p, local_features=n // p),
        range=result,
        normalized_range=normalized,
        lowering=[
            dict(op="local_square_sum", dtype="f16"),
            dict(op="scale", dtype="f32", divisor=n),
            dict(op="sdk_axis_sum_broadcast", axis="y", dtype="f32"),
            dict(op="narrow", dtype="f16"),
            dict(op="normalize", statistic_is_mean=True),
        ],
        status="typed range-checked boundary; parent code generation and SDK qualification required",
    )
