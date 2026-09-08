#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,64>("a");
  #pragma csl dataflow rows=8 cols=4 exchange=neighbors rotation=givens fp=relaxed compute=vector
  auto result = spatial::qr_r(a);
  spatial::output("result", result);
}
