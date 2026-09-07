#include "spatial.hpp"
void design() {
 auto a=spatial::input<64,64,spatial::f16>("a");
 auto b=spatial::input<64,64,spatial::f16>("b");
 #pragma csl dataflow rows=4 cols=4 exchange=two_hop initial_align=bidirectional reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto result=spatial::matmul(a,b);
 spatial::output("result",result);
}
