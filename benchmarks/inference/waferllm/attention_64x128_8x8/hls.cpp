#include "spatial.hpp"
void design() {
 auto q=spatial::input<64,128,spatial::f16>("q");
 auto k=spatial::input<64,128,spatial::f16>("k");
 auto v=spatial::input<64,128,spatial::f16>("v");
 auto kt=spatial::transpose(k);
 #pragma csl dataflow rows=8 cols=8 exchange=vertical_two_hop reduce=rotating_root order=east_first overlap=double_buffer compute=dsr fp=relaxed
 auto score=spatial::matmul(q,kt);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed elementwise=map
 auto probability=spatial::softmax(score,0.08838834764831845);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto output=spatial::matmul(probability,v);
 spatial::output("output",output);
}
