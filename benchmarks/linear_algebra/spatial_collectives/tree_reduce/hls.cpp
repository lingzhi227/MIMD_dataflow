#include "spatial.hpp"
void design() {
auto p0=spatial::input<1,4>("p0");
auto p1=spatial::input<1,4>("p1");
auto p2=spatial::input<1,4>("p2");
auto p3=spatial::input<1,4>("p3");
auto left=spatial::add(p0,p1);auto right=spatial::add(p2,p3);auto result=spatial::add(left,right);spatial::output("result",result);
}
