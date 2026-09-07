from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
catalog = json.loads((ROOT / "catalog.json").read_text())
for dimensions in (1, 2, 3):
    n = 4**dimensions
    lines = n // 4
    index = (
        "line*4+j"
        if dimensions == 1
        else ("line*4+j" if dimensions == 2 else "line*4+j")
    )

    # Each axis enumerates a disjoint set of length-four pencils in row-major data.
    def idx(j):
        if dimensions == 1:
            return f"index={j};"
        if dimensions == 2:
            return f"if(axis==0){{index=line*4+{j};}}else{{index={j}*4+line;}}"
        return f"if(axis==0){{index=line*4+{j};}}else{{if(axis==1){{index=(line/4)*16+{j}*4+line-(line/4)*4;}}else{{index={j}*16+line;}}}}"

    body = f"""spatial::tensor<{n},2> out=x;
for(int axis=0;axis<{dimensions};++axis){{for(int line=0;line<{lines};++line){{
 spatial::tensor<4,2> work{{}};
 for(int j=0;j<4;++j){{int index=0;{idx('j')}int rev=j/2+(j-(j/2)*2)*2;work.data[rev*2]=out.data[index*2];work.data[rev*2+1]=out.data[index*2+1];}}
 for(int pair=0;pair<2;++pair){{int base=pair*4;float ar=work.data[base];float ai=work.data[base+1];float br=work.data[base+2];float bi=work.data[base+3];work.data[base]=ar+br;work.data[base+1]=ai+bi;work.data[base+2]=ar-br;work.data[base+3]=ai-bi;}}
 for(int j=0;j<2;++j){{float ar=work.data[j*2];float ai=work.data[j*2+1];float br=work.data[j*2+4];float bi=work.data[j*2+5];float wr=twiddle.data[j*2];float wi=twiddle.data[j*2+1];float vr=br*wr-bi*wi;float vi=br*wi+bi*wr;work.data[j*2]=ar+vr;work.data[j*2+1]=ai+vi;work.data[j*2+4]=ar-vr;work.data[j*2+5]=ai-vi;}}
 for(int j=0;j<4;++j){{int index=0;{idx('j')}out.data[index*2]=work.data[j*2];out.data[index*2+1]=work.data[j*2+1];}}
}}}}return out;"""
    source = (
        f'#include "spatial.hpp"\nvoid design(){{auto x=spatial::input<{n},2>("x");auto twiddle=spatial::input<2,2>("twiddle");auto result=spatial::kernel(x,twiddle,[](const spatial::tensor<{n},2>& x,const spatial::tensor<2,2>& twiddle){{\n'
        + body
        + '\n});spatial::output("result",result);}\n'
    )
    name = f"fft_{dimensions}d"
    d = ROOT / "projects/sdk_examples" / name
    d.mkdir(exist_ok=True)
    (d / "hls.cpp").write_text(source)
    item = dict(
        project="sdk_examples",
        kernel=name,
        origins=["benchmarks/fft-1d-2d/fft.csl"]
        + (["benchmarks/fft-3d/layout.csl"] if dimensions == 3 else []),
        fixture="fft:" + str(dimensions),
        contract="Length four per axis, radix-2 local pencils and explicit axis traversal; unnormalized forward transform with supplied twiddles. Multi-PE transpose/routing and larger sizes remain pending.",
        partitions=1,
        status="source_ready",
    )
    (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog = [
        x for x in catalog if (x["project"], x["kernel"]) != ("sdk_examples", name)
    ]
    catalog.append(item)
(ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
