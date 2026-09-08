#include "spatial.hpp"
void design() {
  auto a = spatial::input<32,32>("a");
  #pragma csl dataflow rows=4 cols=4 triangle=lower update=right_looking fp=relaxed compute=vector
  auto result = spatial::cholesky(a);
  spatial::output("result", result);
}
