#include "spatial.hpp"
void design(){
auto field=spatial::input<4,4>("field");
auto coeff=spatial::input<1,7>("coeff");
auto halo0=spatial::kernel(field,[](const spatial::tensor<4,4>&f){spatial::tensor<5,4> h{};
for(int k=0;k<4;++k){h.data[0+k]=f.data[0+k];}
for(int k=0;k<4;++k){h.data[8+k]=f.data[8+k];}
for(int k=0;k<4;++k){h.data[16+k]=f.data[4+k];}
return h;});
#pragma csl place x=31 y=4
auto tile0=spatial::kernel(halo0,coeff,[](const spatial::tensor<5,4>&h,const spatial::tensor<1,7>&c){
spatial::tensor<1,4> out{};for(int k=0;k<4;++k){float v=c.data[6]*h.data[k];v+=c.data[0]*h.data[4+k];v+=c.data[1]*h.data[8+k];v+=c.data[2]*h.data[12+k];v+=c.data[3]*h.data[16+k];if(k>0){v+=c.data[4]*h.data[k-1];}if(k<3){v+=c.data[5]*h.data[k+1];}out.data[k]=v;}return out;});
spatial::output("tile0",tile0);
auto halo1=spatial::kernel(field,[](const spatial::tensor<4,4>&f){spatial::tensor<5,4> h{};
for(int k=0;k<4;++k){h.data[0+k]=f.data[4+k];}
for(int k=0;k<4;++k){h.data[8+k]=f.data[12+k];}
for(int k=0;k<4;++k){h.data[12+k]=f.data[0+k];}
return h;});
#pragma csl place x=31 y=7
auto tile1=spatial::kernel(halo1,coeff,[](const spatial::tensor<5,4>&h,const spatial::tensor<1,7>&c){
spatial::tensor<1,4> out{};for(int k=0;k<4;++k){float v=c.data[6]*h.data[k];v+=c.data[0]*h.data[4+k];v+=c.data[1]*h.data[8+k];v+=c.data[2]*h.data[12+k];v+=c.data[3]*h.data[16+k];if(k>0){v+=c.data[4]*h.data[k-1];}if(k<3){v+=c.data[5]*h.data[k+1];}out.data[k]=v;}return out;});
spatial::output("tile1",tile1);
auto halo2=spatial::kernel(field,[](const spatial::tensor<4,4>&f){spatial::tensor<5,4> h{};
for(int k=0;k<4;++k){h.data[0+k]=f.data[8+k];}
for(int k=0;k<4;++k){h.data[4+k]=f.data[0+k];}
for(int k=0;k<4;++k){h.data[16+k]=f.data[12+k];}
return h;});
#pragma csl place x=34 y=4
auto tile2=spatial::kernel(halo2,coeff,[](const spatial::tensor<5,4>&h,const spatial::tensor<1,7>&c){
spatial::tensor<1,4> out{};for(int k=0;k<4;++k){float v=c.data[6]*h.data[k];v+=c.data[0]*h.data[4+k];v+=c.data[1]*h.data[8+k];v+=c.data[2]*h.data[12+k];v+=c.data[3]*h.data[16+k];if(k>0){v+=c.data[4]*h.data[k-1];}if(k<3){v+=c.data[5]*h.data[k+1];}out.data[k]=v;}return out;});
spatial::output("tile2",tile2);
auto halo3=spatial::kernel(field,[](const spatial::tensor<4,4>&f){spatial::tensor<5,4> h{};
for(int k=0;k<4;++k){h.data[0+k]=f.data[12+k];}
for(int k=0;k<4;++k){h.data[4+k]=f.data[4+k];}
for(int k=0;k<4;++k){h.data[12+k]=f.data[8+k];}
return h;});
#pragma csl place x=34 y=7
auto tile3=spatial::kernel(halo3,coeff,[](const spatial::tensor<5,4>&h,const spatial::tensor<1,7>&c){
spatial::tensor<1,4> out{};for(int k=0;k<4;++k){float v=c.data[6]*h.data[k];v+=c.data[0]*h.data[4+k];v+=c.data[1]*h.data[8+k];v+=c.data[2]*h.data[12+k];v+=c.data[3]*h.data[16+k];if(k>0){v+=c.data[4]*h.data[k-1];}if(k<3){v+=c.data[5]*h.data[k+1];}out.data[k]=v;}return out;});
spatial::output("tile3",tile3);
}
