#include "spatial.hpp"
void design(){
 auto values=spatial::input<4608,1>("values");
 auto rows=spatial::index_input<4608,1>("row_indices");
 auto columns=spatial::index_input<513,1>("column_offsets");
 auto initial=spatial::input<512,1>("initial");
 auto steps=spatial::index_input<1,1>("steps");
 #pragma csl dataflow rows=4 cols=4 storage=csc exchange=trains reduce=row_column redistribute=transpose recurrence=resident compute=vector nnz_per_pe=512 cols_per_pe=128 rows_per_pe=128 fp=relaxed
 auto result=spatial::power_csc<512,32>(values,rows,columns,initial,steps);
 spatial::output("vector",result.vector);
 spatial::output("reason",result.reason);
 spatial::output("iterations",result.iterations);
 spatial::output("norms",result.norms);
}
