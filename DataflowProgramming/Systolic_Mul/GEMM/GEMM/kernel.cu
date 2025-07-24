#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    }

#define CHECK_CUBLAS(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int N = 36600;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate host memory for matrices A, B, and C
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);

    // Generate random values for A and B
    generateRandomMatrix(h_A.data(), N, N);
    generateRandomMatrix(h_B.data(), N, N);

    float* d_A, * d_B, * d_C;
    size_t matrixSize = N * N * sizeof(float);

    // Start total CPU timer
    cudaEvent_t cpu_start, cpu_stop;
    CHECK_CUDA(cudaEventCreate(&cpu_start));
    CHECK_CUDA(cudaEventCreate(&cpu_stop));
    CHECK_CUDA(cudaEventRecord(cpu_start, 0));

    // Allocate device memory for matrices A, B, and C
    CHECK_CUDA(cudaMalloc(&d_A, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_B, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_C, matrixSize));

    // Start memcpy timer
    cudaEvent_t memcpy_start, memcpy_stop;
    CHECK_CUDA(cudaEventCreate(&memcpy_start));
    CHECK_CUDA(cudaEventCreate(&memcpy_stop));
    CHECK_CUDA(cudaEventRecord(memcpy_start, 0));

    // Copy matrices A and B to device memory
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), matrixSize, cudaMemcpyHostToDevice));

    // Stop memcpy timer
    CHECK_CUDA(cudaEventRecord(memcpy_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(memcpy_stop));
    float memcpy_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&memcpy_time, memcpy_start, memcpy_stop));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Set cuBLAS to use Tensor Cores
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Start GPU timer
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA(cudaEventCreate(&gpu_start));
    CHECK_CUDA(cudaEventCreate(&gpu_stop));
    CHECK_CUDA(cudaEventRecord(gpu_start, 0));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

    // Stop GPU timer
    CHECK_CUDA(cudaEventRecord(gpu_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(gpu_stop));
    float gpu_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop));

    // Copy result matrix C back to host memory
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost));

    // Stop total CPU timer
    CHECK_CUDA(cudaEventRecord(cpu_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(cpu_stop));
    float total_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_time, cpu_start, cpu_stop));

    // Clean up resources
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(gpu_start));
    CHECK_CUDA(cudaEventDestroy(gpu_stop));
    CHECK_CUDA(cudaEventDestroy(memcpy_start));
    CHECK_CUDA(cudaEventDestroy(memcpy_stop));
    CHECK_CUDA(cudaEventDestroy(cpu_start));
    CHECK_CUDA(cudaEventDestroy(cpu_stop));

    // Output timing results
    std::cout << "Memcpy time: " << memcpy_time << " ms" << std::endl;
    std::cout << "GPU computation time: " << gpu_time << " ms" << std::endl;
    std::cout << "Total time (CPU perspective): " << total_time << " ms" << std::endl;

    return 0;
}
