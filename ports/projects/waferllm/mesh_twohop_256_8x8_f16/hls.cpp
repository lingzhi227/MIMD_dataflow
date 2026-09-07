#include "spatial.hpp"
void design() {
 auto a=spatial::input<256,256,spatial::f16>("a");
 auto b=spatial::input<256,256,spatial::f16>("b");
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=bidirectional reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto result=spatial::matmul(a,b);
 spatial::output("result",result);
}
