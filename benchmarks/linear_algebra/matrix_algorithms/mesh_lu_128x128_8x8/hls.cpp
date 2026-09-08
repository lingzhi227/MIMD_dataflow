#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,128>("a");
  #pragma csl dataflow rows=8 cols=8 pivot=none update=blocked fp=relaxed compute=vector
  auto result = spatial::lu_no_pivot(a);
  spatial::output("result", result);
}
