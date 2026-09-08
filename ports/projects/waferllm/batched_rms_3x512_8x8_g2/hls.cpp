#include "spatial.hpp"
void design() {
    auto x = spatial::input<3,512,spatial::f16>("x",1.0);
    auto w = spatial::input<1,512,spatial::f16>("w",1.0);
#pragma csl dataflow rows=8 cols=8 partition=features axis=y layout=batch_major reduce=grouped_two_tree groups=2 result=replicated_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
    auto normalized = spatial::rmsnorm(x,w,0.000001);
    spatial::output("normalized",normalized);
}
