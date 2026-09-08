"""Correlated RMS L1 range through half blocks, float local merge and SDK reduction."""

import math
from fractions import Fraction as F
from frontend import check
from input_contracts import half_operand_bound
from rms_l1_bounds import U, U32, ETA, upward, half_ceiling, projection


def bound(l1, weight, local_features, participants, block):
    check(
        type(local_features) is int
        and 1 <= local_features <= 512
        and type(participants) is int
        and participants in (4, 8, 16)
        and type(l1) in (int, float, F)
        and math.isfinite(l1)
        and l1 >= 0,
        "blocked projection exact geometry and finite L1",
    )
    check(
        type(block) is int
        and 1 <= block <= local_features
        and local_features % block == 0,
        "projection divisible half blocks",
    )
    if block == local_features:
        return projection(l1, weight, local_features, participants)
    magnitude = F(l1) * F(half_operand_bound(weight))
    gh = (1 + U) ** block
    gl = (1 + U32) ** (local_features // block - 1)
    gg = (1 + U32) ** (participants - 1)
    local = (1 + U) * gl * gh * (magnitude + local_features * ETA / 2) + ETA / 2
    reduced = (1 + U) * gg * (
        (1 + U) * gl * gh * (magnitude + participants * local_features * ETA / 2)
        + participants * ETA / 2
    ) + ETA / 2
    native = (1 + U) * (1 + U32) ** (
        participants * local_features // block - 1
    ) * gh * (magnitude + participants * local_features * ETA / 2) + ETA / 2
    return dict(
        local=half_ceiling(upward(local)),
        reduced=half_ceiling(upward(max(reduced, native))),
        native=half_ceiling(upward(native)),
        block_size=block,
        proof="Global correlated RMS L1 counted once; half block FMA envelope, float local merge, local half narrow, SDK float sum and final half narrow. Native whole-row block merge separately enclosed; exact rational factors.",
    )
