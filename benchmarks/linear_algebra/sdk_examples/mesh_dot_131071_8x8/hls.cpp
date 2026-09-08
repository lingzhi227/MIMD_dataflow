#include "spatial.hpp"
void design(){
 auto x=spatial::input<131071,1>("x");
 auto y=spatial::input<131071,1>("y");
 #pragma csl dataflow rows=8 cols=8 partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map
 auto value=spatial::dot(x,y);
 spatial::output("result",value);
}
