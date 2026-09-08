"""Typed integer boundary for SDK compiler --params; no stringly typed flags.

SDK2.10.1 rejects integer initialization of bool and its command parser can
assert on textual false. Library adapters use an integer layout parameter,
then an explicit CSL boolean conversion inside the layout.
"""

from frontend import check

BOUNDS = {
    "u16": (0, 65535),
    "i16": (-32768, 32767),
    "u32": (0, 4294967295),
    "i32": (-2147483648, 2147483647),
}


def encode(schema, values):
    check(set(schema) == set(values), "compiler parameter schema keys")
    fields = []
    for name, kind in schema.items():
        check(name.isascii() and name.isidentifier(), "compiler parameter identifier")
        check(
            kind in BOUNDS, "external compiler parameter requires supported integer ABI"
        )
        lo, hi = BOUNDS[kind]
        value = values[name]
        check(
            type(value) is int and lo <= value <= hi,
            "compiler parameter " + name + " requires " + kind,
        )
        fields.append(name + ":" + str(value))
    return "--params=" + ",".join(fields)
