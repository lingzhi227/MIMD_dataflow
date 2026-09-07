#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,256>("a");
  auto b = spatial::input<256,128>("b");
  // SUMMA: broadcast A panels by rows and B panels by columns; accumulate locally.
  #pragma csl dataflow rows=8 cols=8 broadcast=rows_columns reduce=local fp=relaxed compute=vector
  auto result = spatial::matmul(a, b);
  spatial::output("result", result);
}
