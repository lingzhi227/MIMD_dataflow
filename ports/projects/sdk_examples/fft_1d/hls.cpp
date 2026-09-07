#include "spatial.hpp"
void design(){auto x=spatial::input<4,2>("x");auto twiddle=spatial::input<2,2>("twiddle");auto result=spatial::kernel(x,twiddle,[](const spatial::tensor<4,2>& x,const spatial::tensor<2,2>& twiddle){
spatial::tensor<4,2> out=x;
for(int axis=0;axis<1;++axis){for(int line=0;line<1;++line){
 spatial::tensor<4,2> work{};
 for(int j=0;j<4;++j){int index=0;index=j;int rev=j/2+(j-(j/2)*2)*2;work.data[rev*2]=out.data[index*2];work.data[rev*2+1]=out.data[index*2+1];}
 for(int pair=0;pair<2;++pair){int base=pair*4;float ar=work.data[base];float ai=work.data[base+1];float br=work.data[base+2];float bi=work.data[base+3];work.data[base]=ar+br;work.data[base+1]=ai+bi;work.data[base+2]=ar-br;work.data[base+3]=ai-bi;}
 for(int j=0;j<2;++j){float ar=work.data[j*2];float ai=work.data[j*2+1];float br=work.data[j*2+4];float bi=work.data[j*2+5];float wr=twiddle.data[j*2];float wi=twiddle.data[j*2+1];float vr=br*wr-bi*wi;float vi=br*wi+bi*wr;work.data[j*2]=ar+vr;work.data[j*2+1]=ai+vi;work.data[j*2+4]=ar-vr;work.data[j*2+5]=ai-vi;}
 for(int j=0;j<4;++j){int index=0;index=j;out.data[index*2]=work.data[j*2];out.data[index*2+1]=work.data[j*2+1];}
}}return out;
});spatial::output("result",result);}
