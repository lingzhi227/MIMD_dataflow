#include "spatial.hpp"
void design() {
    auto a = spatial::input<1,512,spatial::f16>("a");
    auto b = spatial::input<512,512,spatial::f16>("b");
#pragma csl dataflow rows=8 cols=8 broadcast=host_rows reduce=grouped_two_tree groups=4 result=replicated_columns fp=relaxed compute=dsr
    auto result = spatial::matmul(a,b);
    spatial::output("result",result);
}
