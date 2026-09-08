"""Independent stdlib-half oracle for the directed midpoint-tree probe."""

import struct
import numpy as np


def half(value):
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def midpoint(values):
    assert len(values) in (2, 4)
    root = len(values) // 2
    top = values[0]
    for index in range(1, root):
        top = half(top + values[index])
    result = half(values[root] + top)
    if root + 1 < len(values):
        tail = values[-1]
        for index in range(len(values) - 2, root, -1):
            tail = half(tail + values[index])
        result = half(result + tail)
    return result


def axis_words(values, groups, axis):
    values = np.asarray(values)
    assert values.shape[:2] == (8, 8) and groups in (2, 4) and axis in (0, 1)
    output = np.zeros(values.shape, np.uint16)
    size = 8 // groups
    for other in range(8):
        for lane in range(values.shape[2]):
            vector = list(
                values[:, other, lane] if axis == 0 else values[other, :, lane]
            )
            parts = [
                midpoint(vector[start : start + size]) for start in range(0, 8, size)
            ]
            word = struct.unpack("<H", struct.pack("<e", midpoint(parts)))[0]
            if axis == 0:
                output[:, other, lane] = word
            else:
                output[other, :, lane] = word
    return output
