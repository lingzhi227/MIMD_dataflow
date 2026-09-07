#include "spatial.hpp"
void design() {
auto a=spatial::input<4,4>("a");auto b=spatial::input<4,1>("b");auto result=spatial::matmul(a,b);spatial::output("result",result);
}
