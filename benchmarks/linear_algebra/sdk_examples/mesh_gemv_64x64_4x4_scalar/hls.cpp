#include "spatial.hpp"
void design() {
  auto a = spatial::input<64,64>("a");
  auto x = spatial::input<64,1>("x");
  // Partition A across a PE mesh; distribute x by columns and reduce by rows.
  #pragma csl dataflow rows=4 cols=4 broadcast=columns reduce=rows fp=relaxed compute=scalar
  auto result = spatial::matmul(a, x);
  spatial::output("result", result);
}
