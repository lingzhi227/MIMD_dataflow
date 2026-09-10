#include "spatial.hpp"
void design() {
  auto a = spatial::input<64,64>("a");
  auto b = spatial::input<64,64>("b");
  // SUMMA: broadcast A panels by rows and B panels by columns; accumulate locally.
  #pragma csl dataflow rows=4 cols=4 broadcast=rows_columns reduce=local fp=relaxed compute=vector
  auto result = spatial::matmul(a, b);
  spatial::output("result", result);
}
