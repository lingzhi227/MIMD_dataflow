#include "spatial.hpp"
void design() {
  auto x=spatial::input<2,4>("x");
  auto coeff=spatial::input<1,4>("coeff");
  auto result=spatial::kernel(x,coeff, [](const spatial::tensor<2,4>& x, const spatial::tensor<1,4>& coeff) {
spatial::tensor<2,4> out{};for(int i=0;i<2;++i){for(int pair=0;pair<2;++pair){float even=x.data[i*4+pair*2];float odd=x.data[i*4+pair*2+1];float cs=coeff.data[pair*2];float sn=coeff.data[pair*2+1];out.data[i*4+pair*2]=even*cs-odd*sn;out.data[i*4+pair*2+1]=odd*cs+even*sn;}}return out;
  });
  spatial::output("result",result);
}
