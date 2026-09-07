#include "spatial.hpp"
void design(){
 auto x=spatial::input<8191,1>("x");
 auto y=spatial::input<8191,1>("y");
 #pragma csl dataflow rows=4 cols=4 partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map
 auto value=spatial::dot(x,y);
 spatial::output("result",value);
}
