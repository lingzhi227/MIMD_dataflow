#include "spatial.hpp"
void design() {
  auto a = spatial::input<256,256>("a");
  auto x = spatial::input<256,1>("x");
  // Partition A across a PE mesh; distribute x by columns and reduce by rows.
  #pragma csl dataflow rows=8 cols=8 broadcast=columns reduce=rows fp=relaxed compute=vector
  auto result = spatial::matmul(a, x);
  spatial::output("result", result);
}
