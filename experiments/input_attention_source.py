"""Source-shaped31-node C++ authoring for the unqualified diagnostic lowering."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from build_attention_tail import source as attention_source


def source(m=64, n=64, f=256, p=8):
    text = attention_source(m, n, f, p)
    for key in ("q", "k", "v"):
        old = f' auto {key}=spatial::input<{m},{n},spatial::f16>("{key}",0.125);'
        assert text.count(old) == 1
        text = text.replace(
            old,
            f' auto {key}_weight=spatial::input<{n},{n},spatial::f16>("{key}_weight",0.00390625);',
        )
    text = text.replace(
        f' auto residual=spatial::input<{m},{n},spatial::f16>("residual",0.125);',
        f' auto input_x=spatial::input<{m},{n},spatial::f16>("input_x",0.125);',
    )
    text = text.replace(
        "spatial::add(projection,residual)", "spatial::add(projection,input_x)"
    )
    marker = " auto kt=spatial::transpose(k);"
    assert text.count(marker) == 1
    prefix = f""" auto cosine=spatial::input<1,{n//2},spatial::f16>("cosine",1.0);
 auto sine=spatial::input<1,{n//2},spatial::f16>("sine",1.0);
 #pragma csl dataflow rows={p} cols={p} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto input_normalized=spatial::rmsnorm(input_x,gamma,0.000001);
"""
    for key in ("q", "k", "v"):
        name = key + "_raw" if key != "v" else "v"
        prefix += f""" #pragma csl dataflow rows={p} cols={p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto {name}=spatial::matmul(input_normalized,{key}_weight);
"""
    for key in ("q", "k"):
        prefix += f""" #pragma csl dataflow rows={p} cols={p} partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto {key}=spatial::rotate_pairs<spatial::pair_order::odd_even>({key}_raw,cosine,sine);
"""
    return text.replace(marker, prefix + marker)
