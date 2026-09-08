#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,128>("a");
  #pragma csl dataflow rows=8 cols=8 triangle=lower update=right_looking fp=relaxed compute=vector
  auto result = spatial::cholesky(a);
  spatial::output("result", result);
}
