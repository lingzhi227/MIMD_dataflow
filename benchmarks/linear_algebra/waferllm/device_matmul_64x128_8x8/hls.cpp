#include "spatial.hpp"
void design() {
 auto a=spatial::input<64,64,spatial::f16>("a");
 auto b=spatial::input<64,128,spatial::f16>("b");
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto product=spatial::matmul(a,b);
 spatial::output("product",product);
}
