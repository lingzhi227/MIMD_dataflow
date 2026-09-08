#include "spatial.hpp"
void design() {
  auto a = spatial::input<32,32>("a");
  #pragma csl dataflow rows=4 cols=4 exchange=neighbors rotation=givens fp=relaxed compute=vector
  auto result = spatial::qr_r(a);
  spatial::output("result", result);
}
