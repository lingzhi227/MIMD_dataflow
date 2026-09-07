#include "spatial.hpp"
void design() {
  auto a = spatial::input<4, 4>("a");
  auto l = spatial::kernel(a, [](const spatial::tensor<4, 4> &a) {
    spatial::tensor<4, 4> l = a;
    for (int k = 0; k < 4; ++k) {
      spatial::require(l.data[k * 4 + k] > 0.0f);
      l.data[k * 4 + k] = spatial::sqrt(l.data[k * 4 + k]);
      for (int i = k + 1; i < 4; ++i) {
        l.data[i * 4 + k] /= l.data[k * 4 + k];
      }
      for (int j = k + 1; j < 4; ++j) {
        for (int i = j; i < 4; ++i) {
          l.data[i * 4 + j] -= l.data[i * 4 + k] * l.data[j * 4 + k];
        }
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        l.data[i * 4 + j] = 0.0f;
      }
    }
    return l;
  });
  spatial::output("l", l);
}
