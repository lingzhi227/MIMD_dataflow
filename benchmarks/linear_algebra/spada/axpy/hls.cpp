#include "spatial.hpp"
void design() {
auto x=spatial::input<1,4>("x");auto y=spatial::input<1,4>("y");auto ax=spatial::map(x,[](float x){return 2.5f*x;});auto result=spatial::add(ax,y);spatial::output("result",result);
}
