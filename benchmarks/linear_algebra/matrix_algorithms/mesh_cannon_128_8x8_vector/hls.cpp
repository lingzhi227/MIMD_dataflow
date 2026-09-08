#include "spatial.hpp"
void design() {
 auto a = spatial::input<128,128>("a");
 auto b = spatial::input<128,128>("b");
 #pragma csl dataflow rows=8 cols=8 exchange=cyclic initial_align=host reduce=local fp=relaxed compute=vector
 auto c = spatial::matmul(a,b);
 spatial::output("result",c);
}
