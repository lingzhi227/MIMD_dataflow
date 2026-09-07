#include "spatial.hpp"
void design() {
auto p0=spatial::input<1,4>("p0");
auto p1=spatial::input<1,4>("p1");
auto p2=spatial::input<1,4>("p2");
auto p3=spatial::input<1,4>("p3");
auto s2=spatial::add(p2,p3);auto s1=spatial::add(p1,s2);auto result=spatial::add(p0,s1);spatial::output("result",result);
}
