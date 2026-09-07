"""Experimental SDK exp source model; conversion policy awaits exhaustive probe.

Kept out of the compiler until observed range-reduction and exp bits qualify it.
"""

import math, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from binary16 import quantize
from float32 import f32


def f32word(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def halfword(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def model(x, conversion="nearest"):
    if x < halfword(0xCC55):
        return 0.0, 0, 0.0
    if x > halfword(0x498C):
        return math.inf, 0, 0.0
    scaled = f32(x * f32word(0x40B8AA3B))
    n = round(scaled) if conversion == "nearest" else math.trunc(scaled)
    remainder = quantize(f32(x + f32(n * f32word(0xBE317218))))
    halfpoly = quantize(1 + quantize(0.5 * remainder))
    poly = f32(1 + f32(halfpoly * remainder))
    j = n & 3
    m = (n - j) >> 2
    table = [0x3F800000, 0x3F9837F0, 0x3FB504F3, 0x3FD744FD]
    expn = f32(math.ldexp(f32word(table[j]), m))
    return quantize(f32(expn * poly)), n, remainder
