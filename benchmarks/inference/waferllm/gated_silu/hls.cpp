#include "spatial.hpp"
void design() {
  auto up=spatial::input<2,4>("up");
  auto gate=spatial::input<2,4>("gate");
  auto result=spatial::kernel(up,gate, [](const spatial::tensor<2,4>& up, const spatial::tensor<2,4>& gate) {
spatial::tensor<2,4> out{};for(int i=0;i<8;++i){out.data[i]=up.data[i]*gate.data[i]/(1.0f+spatial::exp(-gate.data[i]));}return out;
  });
  spatial::output("result",result);
}
