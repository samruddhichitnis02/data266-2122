

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#define TILE_SIZE 16

__global__ void matrixMultiply(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void cpuMatrixMultiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N
) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

int main() {
    std::vector<int> matrix_sizes = {256, 1024, 4096};

    std::cout << "Matrix Size,CPU Time (ms),GPU Kernel Time (ms),"
              << "Transfer Time (ms),GPU End-to-End Time (ms),Speedup"
              << std::endl;

    for (int N : matrix_sizes) {
        size_t number_of_elements = static_cast<size_t>(N) * N;
        size_t bytes = number_of_elements * sizeof(float);

        std::vector<float> A(number_of_elements, 1.0f);
        std::vector<float> B(number_of_elements, 1.0f);
        std::vector<float> C_cpu(number_of_elements, 0.0f);
        std::vector<float> C_gpu(number_of_elements, 0.0f);

        // CPU timing
        auto cpu_start = std::chrono::high_resolution_clock::now();

        cpuMatrixMultiply(A, B, C_cpu, N);

        auto cpu_end = std::chrono::high_resolution_clock::now();

        double cpu_time = std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();

        float* d_A;
        float* d_B;
        float* d_C;

        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaEvent_t start_total, stop_total;
        cudaEvent_t start_kernel, stop_kernel;

        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        // Measure total GPU time: transfers plus kernel
        cudaEventRecord(start_total);

        cudaMemcpy(
            d_A, A.data(), bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            d_B, B.data(), bytes,
            cudaMemcpyHostToDevice
        );

        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

        dim3 numberOfBlocks(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        cudaEventRecord(start_kernel);

        matrixMultiply<<<numberOfBlocks, threadsPerBlock>>>(
            d_A, d_B, d_C, N
        );

        cudaEventRecord(stop_kernel);

        cudaMemcpy(
            C_gpu.data(), d_C, bytes,
            cudaMemcpyDeviceToHost
        );

        cudaEventRecord(stop_total);
        cudaEventSynchronize(stop_total);

        float total_gpu_time = 0.0f;
        float kernel_time = 0.0f;

        cudaEventElapsedTime(
            &total_gpu_time,
            start_total,
            stop_total
        );

        cudaEventElapsedTime(
            &kernel_time,
            start_kernel,
            stop_kernel
        );

        float transfer_time = total_gpu_time - kernel_time;
        double speedup = cpu_time / total_gpu_time;

        std::cout << std::fixed << std::setprecision(3)
                  << N << ","
                  << cpu_time << ","
                  << kernel_time << ","
                  << transfer_time << ","
                  << total_gpu_time << ","
                  << speedup
                  << std::endl;

        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
