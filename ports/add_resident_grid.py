"""Configure the same high-level seven-point update at several actual grid scales."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
catalog = json.loads((ROOT / "catalog.json").read_text())
for x, y, z, steps in [(1, 1, 7, 3), (2, 2, 32, 4), (4, 4, 128, 16), (8, 8, 128, 16)]:
    name = f"resident_stencil_{x}x{y}x{z}_t{steps}"
    d = ROOT / "projects/sdk_examples" / name
    d.mkdir(exist_ok=True)
    source = f"""#include "spatial.hpp"
void design(){{
 auto field=spatial::input<{x*y},{z}>("field");
 auto coeff=spatial::input<1,7>("coeff");
 {"#pragma csl vectorize" if x>1 else ""}
 auto result=spatial::grid_iterate<{x},{y},{z},{steps}>(field,coeff,[](const spatial::tensor<7,{z}>&n,const spatial::tensor<1,7>&c){{
  spatial::tensor<1,{z}> out{{}};
  for(int k=0;k<{z};++k){{
   float value=c.data[6]*n.data[{6*z}+k];
   for(int direction=0;direction<6;++direction){{value+=c.data[direction]*n.data[direction*{z}+k];}}
   out.data[k]=value;
  }}return out;
 }});
 spatial::output("result",result);
}}
"""
    item = dict(
        project="sdk_examples",
        kernel=name,
        origins=[
            "benchmarks/7pt-stencil-spmv/src/kernel.csl",
            "benchmarks/benchmark-libs/stencil_3d_7pts/wse3/pe.csl",
        ],
        fixture=f"grid:{x}:{y}:{z}:{steps}",
        contract="SDK seven-point operator iterated on resident pencils; zero exterior; four-neighbor on-device exchange with asynchronous full-frame DSD sends. Optional checked DSD vector arithmetic preserves multiply/add ordering; original block-overlapped DSR/fmac implementation not yet performance-matched.",
        partitions=1,
        status="source_ready",
    )
    (d / "hls.cpp").write_text(source)
    (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog = [
        i for i in catalog if (i["project"], i["kernel"]) != ("sdk_examples", name)
    ] + [item]
(ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
