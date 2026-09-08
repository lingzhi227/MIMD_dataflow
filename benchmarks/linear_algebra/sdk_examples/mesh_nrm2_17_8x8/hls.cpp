#include "spatial.hpp"
void design(){
 auto x=spatial::input<17,1>("x");
 #pragma csl dataflow rows=8 cols=8 partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map
 auto value=spatial::nrm2(x);
 spatial::output("result",value);
}
