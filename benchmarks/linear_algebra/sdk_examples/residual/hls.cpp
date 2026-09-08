#include "spatial.hpp"
void design() {
auto a=spatial::input<4,4>("a");auto x=spatial::input<4,1>("x");auto b=spatial::input<4,1>("b");auto ax=spatial::matmul(a,x);auto negative=spatial::map(ax,[](float x){return -x;});auto result=spatial::add(b,negative);spatial::output("result",result);
}
