#include "spatial.hpp"
void design() {
  auto x=spatial::input<2,4>("x");
  auto w=spatial::input<1,4>("w");
  auto result=spatial::kernel(x,w, [](const spatial::tensor<2,4>& x, const spatial::tensor<1,4>& w) {
spatial::tensor<2,4> out{};for(int i=0;i<2;++i){float sum=0.0f;for(int j=0;j<4;++j){sum+=x.data[i*4+j]*x.data[i*4+j];}float inv=1.0f/spatial::sqrt(sum/4.0f+0.000001f);for(int j=0;j<4;++j){out.data[i*4+j]=x.data[i*4+j]*inv*w.data[j];}}return out;
  });
  spatial::output("result",result);
}
