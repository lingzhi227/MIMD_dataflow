#include "spatial.hpp"
void design() {
 auto x=spatial::input<64,64,spatial::f16>("x");
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto probability=spatial::softmax(x,0.125);
 spatial::output("probability",probability);
}
