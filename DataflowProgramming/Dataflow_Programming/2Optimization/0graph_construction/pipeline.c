// gcc -fopenmp pipeline.c -o pipeline
// ./pipeline

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void process_data_pipeline(int input_data[], int output_data[], int n) {
    int *intermediate1 = (int*)malloc(n * sizeof(int));
    int *intermediate2 = (int*)malloc(n * sizeof(int));
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Process data elements one by one through the pipeline
            for (int i = 0; i < n; i++) {
                // Node 1: Square operation
                #pragma omp task depend(out: intermediate1[i])
                {
                    intermediate1[i] = input_data[i] * input_data[i];
                    printf("Thread %d: Node 1 squared data[%d] = %d to get %d\n", 
                           omp_get_thread_num(), i, input_data[i], intermediate1[i]);
                }
                
                // Node 2: Double operation
                #pragma omp task depend(in: intermediate1[i]) depend(out: intermediate2[i])
                {
                    intermediate2[i] = intermediate1[i] * 2;
                    printf("Thread %d: Node 2 doubled data[%d] = %d to get %d\n", 
                           omp_get_thread_num(), i, intermediate1[i], intermediate2[i]);
                }
                
                // Node 3: Add operation (adding the original value to the doubled-square)
                #pragma omp task depend(in: intermediate2[i]) depend(out: output_data[i])
                {
                    output_data[i] = intermediate2[i] + input_data[i];
                    printf("Thread %d: Node 3 added data[%d] = %d to %d to get %d\n", 
                           omp_get_thread_num(), i, input_data[i], intermediate2[i], output_data[i]);
                }
            }
        }
    }
    
    free(intermediate1);
    free(intermediate2);
}

int main() {
    int n = 5;
    int input_data[5] = {1, 2, 3, 4, 5};
    int output_data[5];
    
    printf("Starting pipeline processing with %d data elements\n", n);
    
    process_data_pipeline(input_data, output_data, n);
    
    printf("\nFinal results:\n");
    for (int i = 0; i < n; i++) {
        // output = input + 2*(input^2)
        printf("data[%d] = %d → result = %d\n", i, input_data[i], output_data[i]);
    }
    
    return 0;
}