from source_tree import source_path, logical_name
"""Add checked extent selection to the single-instance source grouped collective.

Keep the qualified static source module unchanged. This lowering varies DSD
lengths, never routes/queues. Every rank must call collectives in the same order;
local blocking completion and FIFO ordering protect transitions between lengths.
"""

import re
from pathlib import Path
from frontend import check


def generate(dest):
    source = (source_path("runtime/axis_grouped_reduce.csl")).read_text()
    names = re.findall(r"const (\w+) = @get_dsd\((?:fabin|fabout)_dsd", source)
    check(len(names) == 10, "source grouped collective has ten directional fabric DSDs")
    for name in names:
        source = source.replace(
            "const " + name + " = @get_dsd", "var " + name + " = @get_dsd", 1
        )
    source += """
// Select a padded vector length only after the preceding local collective call
// has returned. Same Y routes and ordered FIFO streams are retained.
fn set_extent(length:u16) void {
 @assert(length>0 and length%2==0 and length<=@as(u16,bsz));
"""
    for name in names + ["vector_buf_dsd_bsz"]:
        source += f" {name}=@set_dsd_length({name},length);\n"
    source += "}\n"
    (Path(dest) / "axis_grouped_reduce_dynamic.csl").write_text(source)
