#include "spatial.hpp"
void design(){
 auto x=spatial::input<8191,1>("x");
 #pragma csl dataflow rows=4 cols=4 partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map
 auto value=spatial::nrm2(x);
 spatial::output("result",value);
}
