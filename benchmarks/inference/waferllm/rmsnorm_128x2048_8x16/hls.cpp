#include "spatial.hpp"
void design() {
 auto x=spatial::input<128,2048,spatial::f16>("x");
 auto w=spatial::input<1,2048,spatial::f16>("w");
 #pragma csl dataflow rows=8 cols=16 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
 spatial::output("normalized",normalized);
}
