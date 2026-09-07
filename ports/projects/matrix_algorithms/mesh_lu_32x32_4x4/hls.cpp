#include "spatial.hpp"
void design() {
  auto a = spatial::input<32,32>("a");
  #pragma csl dataflow rows=4 cols=4 pivot=none update=blocked fp=relaxed compute=vector
  auto result = spatial::lu_no_pivot(a);
  spatial::output("result", result);
}
