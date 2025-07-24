#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Custom directives to indicate the start and end of CSL inline code
// These macros help the translation tool recognize CSL code blocks during preprocessing
#define CSL_BEGIN(kernel_name) /* <CSL_KERNEL name=kernel_name> */
#define CSL_END /* </CSL_KERNEL> */

void process_data_pipeline(int input_data[], int output_data[], int n) {
    int *intermediate1 = (int*)malloc(n * sizeof(int));
    int *intermediate2 = (int*)malloc(n * sizeof(int));
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Process data elements through the pipeline
            for (int i = 0; i < n; i++) {
                // Node 1: Square operation
                #pragma omp task depend(out: intermediate1[i]) csl_kernel(square_kernel)
                {
                    // Regular C code - ignored during translation
                    intermediate1[i] = input_data[i] * input_data[i];
                    
                    // CSL inline code - extracted for generating the corresponding CSL kernel
                    CSL_BEGIN(square_kernel)
                    kernel void square_kernel(
                        dram_in int input,
                        dram_out int output
                    ) {
                        output = input * input;
                    }
                    CSL_END
                }
                
                // Node 2: Doubling operation
                #pragma omp task depend(in: intermediate1[i]) depend(out: intermediate2[i]) csl_kernel(double_kernel)
                {
                    // Regular C code
                    intermediate2[i] = intermediate1[i] * 2;
                    
                    // CSL inline code
                    CSL_BEGIN(double_kernel)
                    kernel void double_kernel(
                        dram_in int input,
                        dram_out int output
                    ) {
                        output = input * 2;
                    }
                    CSL_END
                }
                
                // Node 3: Addition operation
                #pragma omp task depend(in: intermediate2[i]) depend(out: output_data[i]) csl_kernel(add_kernel)
                {
                    // Regular C code
                    output_data[i] = intermediate2[i] + input_data[i];
                    
                    // CSL inline code
                    CSL_BEGIN(add_kernel)
                    kernel void add_kernel(
                        dram_in int intermediate,
                        dram_in int original,
                        dram_out int output
                    ) {
                        output = intermediate + original;
                    }
                    CSL_END
                }
            }
        }
    }
    
    free(intermediate1);
    free(intermediate2);
}

// Main function for demonstration
int main() {
    int n = 5;
    int input_data[5] = {1, 2, 3, 4, 5};
    int output_data[5];
    
    printf("Starting pipeline processing for %d data elements\n", n);
    
    process_data_pipeline(input_data, output_data, n);
    
    printf("\nFinal results:\n");
    for (int i = 0; i < n; i++) {
        printf("data[%d] = %d → result = %d\n", i, input_data[i], output_data[i]);
    }
    
    return 0;
}
