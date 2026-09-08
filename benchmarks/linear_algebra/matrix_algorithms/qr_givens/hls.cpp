#include "spatial.hpp"
void design() {
  auto a=spatial::input<4,4>("a");
  auto result=spatial::kernel(a, [](const spatial::tensor<4,4>& a) {
spatial::tensor<8,4> out{};
for(int i=0;i<4;++i){for(int j=0;j<4;++j){out.data[i*4+j]=a.data[i*4+j];if(i==j){out.data[16+i*4+j]=1.0f;}}}
for(int k=0;k<4;++k){for(int i=k+1;i<4;++i){
 float av=out.data[k*4+k];float bv=out.data[i*4+k];float cs=1.0f;float sn=0.0f;
 if(bv!=0.0f){if(spatial::abs(bv)>spatial::abs(av)){float tau=-av/bv;sn=1.0f/spatial::sqrt(1.0f+tau*tau);cs=sn*tau;}else{float tau=-bv/av;cs=1.0f/spatial::sqrt(1.0f+tau*tau);sn=cs*tau;}}
 for(int j=0;j<4;++j){float top=out.data[k*4+j];float bottom=out.data[i*4+j];out.data[k*4+j]=cs*top-sn*bottom;out.data[i*4+j]=sn*top+cs*bottom;
 float qt=out.data[16+k*4+j];float qb=out.data[16+i*4+j];out.data[16+k*4+j]=cs*qt-sn*qb;out.data[16+i*4+j]=sn*qt+cs*qb;}
}}
return out;
  });
  spatial::output("result",result);
}
