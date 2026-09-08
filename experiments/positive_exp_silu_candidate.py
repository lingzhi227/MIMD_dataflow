"""Experimental compound half model; keep outside production pending SDK evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import math
from half_exp_candidate import model
from binary16 import quantize
from sdk_math_reference import exp_f16_nonpositive


def half(value):
    try:
        return quantize(value)
    except OverflowError:
        return math.copysign(math.inf, value)


def evaluate(magnitude):
    # The existing source-derived exp candidate already carries the positive
    # range guard; explicitly model the final cast overflow at its boundary.
    try:
        exponent = model(magnitude)[0]
    except OverflowError:
        exponent = math.inf
    small = exp_f16_nonpositive(-magnitude)
    negative = half(-magnitude / half(1 + exponent))
    stable_negative = half(half(-magnitude * small) / half(1 + small))
    positive = half(magnitude / half(1 + small))
    return exponent, negative, stable_negative, positive
