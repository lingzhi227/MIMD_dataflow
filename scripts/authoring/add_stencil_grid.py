"""Author a four-pencil HLS graph; this is not part of the compiler."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json
from pathlib import Path

ROOT = repository_root(__file__)
item = dict(
    project="sdk_examples",
    kernel="stencil_7pt_grid",
    origins=[
        "benchmarks/7pt-stencil-spmv/src/kernel.csl",
        "benchmarks/benchmark-libs/stencil_3d_7pts/wse3/pe.csl",
    ],
    fixture="stencil_grid",
    contract="Source seven-point operator on a 2x2x4 grid, zero exterior values, four spatial compute actors. Device actors gather pencil neighborhoods; host supplies only the global field and coefficients. Generic graph fanout/gather replaces original streaming nearest-neighbor exchange; no temporal loop or performance equivalence claimed.",
    partitions=1,
    status="source_ready",
)
lines = [
    '#include "spatial.hpp"',
    "void design(){",
    'auto field=spatial::input<4,4>("field");',
    'auto coeff=spatial::input<1,7>("coeff");',
]
for x in range(2):
    for y in range(2):
        t = x * 2 + y
        lines.append(
            f"auto halo{t}=spatial::kernel(field,[](const spatial::tensor<4,4>&f){{spatial::tensor<5,4> h{{}};"
        )
        for slot, (i, j) in enumerate(
            [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        ):
            if 0 <= i < 2 and 0 <= j < 2:
                lines.append(
                    f"for(int k=0;k<4;++k){{h.data[{slot*4}+k]=f.data[{(i*2+j)*4}+k];}}"
                )
        lines.extend(
            [
                "return h;});",
                f"#pragma csl place x={31+3*x} y={4+3*y}",
                f"auto tile{t}=spatial::kernel(halo{t},coeff,[](const spatial::tensor<5,4>&h,const spatial::tensor<1,7>&c){{",
                "spatial::tensor<1,4> out{};for(int k=0;k<4;++k){float v=c.data[6]*h.data[k];v+=c.data[0]*h.data[4+k];v+=c.data[1]*h.data[8+k];v+=c.data[2]*h.data[12+k];v+=c.data[3]*h.data[16+k];if(k>0){v+=c.data[4]*h.data[k-1];}if(k<3){v+=c.data[5]*h.data[k+1];}out.data[k]=v;}return out;});",
                f'spatial::output("tile{t}",tile{t});',
            ]
        )
lines.append("}")
d = ROOT / "benchmarks" / item["project"] / item["kernel"]
d.mkdir(exist_ok=True)
(d / "hls.cpp").write_text("\n".join(lines) + "\n")
(d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
catalog = json.loads((ROOT / "benchmarks/catalog.json").read_text())
catalog = [
    v
    for v in catalog
    if (v["project"], v["kernel"]) != (item["project"], item["kernel"])
] + [item]
(ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
