#include "spatial.hpp"
void design() {
    auto a = spatial::input<1,128,spatial::f16>("a");
    auto b = spatial::input<128,128,spatial::f16>("b");
#pragma csl dataflow rows=4 cols=4 broadcast=host_rows reduce=grouped_two_tree groups=2 result=replicated_columns fp=relaxed compute=dsr
    auto result = spatial::matmul(a,b);
    spatial::output("result",result);
}
