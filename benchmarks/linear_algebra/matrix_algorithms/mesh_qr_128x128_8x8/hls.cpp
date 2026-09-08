#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,128>("a");
  #pragma csl dataflow rows=8 cols=8 exchange=neighbors rotation=givens fp=relaxed compute=vector
  auto result = spatial::qr_r(a);
  spatial::output("result", result);
}
