"""Declarative syntax contracts; ordering is independent of spatial semantics.

This parser only recognizes supported directives and attribute types. Shape,
floating-point policy compatibility and target resource checks remain in IR
verification and lowering. Unknown or duplicate attributes fail closed.
"""

import re

DATAFLOW_SCHEMAS = tuple(
    dict(token.split("=", 1) for token in spec.split())
    for spec in (
        "rows=uint cols=uint exchange=two_hop initial_align=forward|both_axes reduce=local overlap=double_buffer fp=relaxed compute=dsr accumulation=f32",
        "rows=uint cols=uint partition=tiles reduce=max_sum accumulation=f32 math=sdk_float compute=dsr fp=relaxed elementwise=map|scalar",
        "rows=uint cols=uint partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f32 math=sdk_float compute=dsr fp=relaxed",
        "rows=uint cols=uint partition=tiles compute=dsr fp=relaxed",
        "rows=uint cols=uint exchange=vertical_two_hop reduce=rotating_root order=east_first overlap=double_buffer compute=dsr fp=relaxed",
        "rows=uint cols=uint partition=tiles coefficients=feature_pairs|per_token compute=dsd fp=relaxed",
        "rows=uint cols=uint partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed",
        "rows=uint cols=uint partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed elementwise=map|scalar",
        "rows=uint cols=uint partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed",
        "rows=uint cols=uint partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed",
        "rows=uint cols=uint partition=pencils exchange=sdk_transpose compute=sdk_fft result=input_layout|transposed_pencils fp=relaxed",
        "rows=uint cols=uint storage=csc exchange=trains reduce=row_column redistribute=transpose recurrence=resident compute=vector nnz_per_pe=uint cols_per_pe=uint rows_per_pe=uint fp=relaxed",
        "rows=uint cols=uint partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map",
        "rows=uint cols=uint storage=csc exchange=trains reduce=sparse_rows nnz_per_pe=uint cols_per_pe=uint rows_per_pe=uint fp=relaxed",
        "rows=uint cols=uint exchange=neighbors rotation=givens fp=relaxed compute=vector",
        "rows=uint cols=uint pivot=none update=blocked fp=relaxed compute=vector",
        "rows=uint cols=uint triangle=lower update=right_looking fp=relaxed compute=vector",
        "rows=uint cols=uint broadcast=host_rows reduce=grouped_two_tree groups=uint result=replicated_columns fp=relaxed compute=dsr",
        "rows=uint cols=uint exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr accumulation=block_f32",
        "rows=uint cols=uint exchange=two_hop initial_align=bidirectional|forward|both_axes reduce=local overlap=double_buffer fp=relaxed compute=dsr",
        "rows=uint cols=uint exchange=cyclic initial_align=host reduce=local fp=relaxed compute=vector|scalar",
        "rows=uint cols=uint broadcast=columns|rows_columns reduce=rows|local fp=relaxed compute=vector|scalar",
    )
)
INTEGER_ATTRIBUTES = frozenset(
    k for schema in DATAFLOW_SCHEMAS for k, v in schema.items() if v == "uint"
)
_ATTRIBUTE = re.compile(
    r"\s*([a-z_][a-z0-9_]*)\s*=\s*([a-z_][a-z0-9_]*|[0-9]+)(?=\s|$)"
)


def attributes(text):
    result = {}
    position = 0
    text = text.strip()
    while position < len(text):
        m = _ATTRIBUTE.match(text, position)
        if m is None:
            raise ValueError("expected key=value near " + repr(text[position:]))
        key, value = m.groups()
        if key in result:
            raise ValueError("duplicate attribute " + key)
        result[key] = value
        position = m.end()
    return result


def parse(line):
    m = re.fullmatch(r"\s*#pragma\s+csl\s+([a-z_]+)(?:\s+(.*?))?\s*", line)
    if m is None:
        raise ValueError("expected #pragma csl directive")
    directive, text = m.groups()
    text = text or ""
    if directive in ("resident", "vectorize"):
        if text:
            raise ValueError(directive + " takes no attributes")
        return directive
    a = attributes(text)
    if directive == "place":
        if set(a) != {"x", "y"} or not all(v.isdecimal() for v in a.values()):
            raise ValueError("place requires unsigned x and y")
        return f"place x={a['x']} y={a['y']}"
    if directive != "dataflow":
        raise ValueError("unknown CSL directive " + directive)
    for schema in DATAFLOW_SCHEMAS:
        if set(a) == set(schema) and all(
            a[k].isdecimal() if rule == "uint" else a[k] in rule.split("|")
            for k, rule in schema.items()
        ):
            # Canonical field order keeps existing staged IR reproducible.
            return "dataflow " + " ".join(k + "=" + a[k] for k in schema)
    raise ValueError(
        "unsupported dataflow attributes or values: " + ", ".join(sorted(a))
    )
