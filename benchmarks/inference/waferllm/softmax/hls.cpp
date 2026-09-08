#include "spatial.hpp"
void design() {
  auto x=spatial::input<2,4>("x");
  auto result=spatial::kernel(x, [](const spatial::tensor<2,4>& x) {
spatial::tensor<2,4> out{};for(int i=0;i<2;++i){float largest=x.data[i*4];for(int j=1;j<4;++j){if(x.data[i*4+j]>largest){largest=x.data[i*4+j];}}float sum=0.0f;for(int j=0;j<4;++j){out.data[i*4+j]=spatial::exp(x.data[i*4+j]-largest);sum+=out.data[i*4+j];}for(int j=0;j<4;++j){out.data[i*4+j]/=sum;}}return out;
  });
  spatial::output("result",result);
}
