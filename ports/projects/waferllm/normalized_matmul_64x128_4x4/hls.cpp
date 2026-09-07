#include "spatial.hpp"
void design() {
 auto x=spatial::input<64,128,spatial::f16>("x");
 auto w=spatial::input<1,128,spatial::f16>("w");
 #pragma csl dataflow rows=4 cols=4 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
 auto q=spatial::input<128,128,spatial::f16>("q");
 #pragma csl dataflow rows=4 cols=4 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projected=spatial::matmul(normalized,q);
 spatial::output("projected",projected);
}
