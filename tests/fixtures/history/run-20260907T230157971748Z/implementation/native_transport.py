"""Decode the native executable's typed, sequential output protocol."""

from frontend import check
from float32 import f32


def parse_outputs(text):
    outputs = []
    for line in text.splitlines():
        words = line.split()
        check(bool(words), "empty native output line")
        if words[0] == "epoch":
            check(
                len(words) == 2 and words[1] == str(len(outputs)), "native epoch order"
            )
            outputs.append({})
            continue
        check(bool(outputs), "native output before epoch")
        integer = words[0] == "@u32"
        offset = int(integer)
        check(len(words) >= offset + 2, "native output header")
        name = words[offset]
        check(name not in outputs[-1], "duplicate native output")
        count = words[offset + 1]
        raw = words[offset + 2 :]
        check(
            count.isascii() and count.isdigit() and int(count) == len(raw),
            "native output extent",
        )
        if integer:
            check(
                all(v.isascii() and v.isdigit() and int(v) <= 4294967295 for v in raw),
                "native u32 output range",
            )
            values = list(map(int, raw))
        else:
            try:
                # The executable prints binary32 with max_digits10=9. Recover
                # that type before exact application checks, not decimal f64.
                values = [f32(float(value)) for value in raw]
            except ValueError:
                check(False, "native finite binary32 output")
        outputs[-1][name] = values
    check(bool(outputs) and all(outputs), "native output epochs must contain ports")
    return outputs
