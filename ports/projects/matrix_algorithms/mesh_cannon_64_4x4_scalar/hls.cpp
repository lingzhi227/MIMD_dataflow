#include "spatial.hpp"
void design() {
 auto a = spatial::input<64,64>("a");
 auto b = spatial::input<64,64>("b");
 #pragma csl dataflow rows=4 cols=4 exchange=cyclic initial_align=host reduce=local fp=relaxed compute=scalar
 auto c = spatial::matmul(a,b);
 spatial::output("result",c);
}
