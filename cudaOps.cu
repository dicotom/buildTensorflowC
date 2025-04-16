// --- START OF FILE cudaOps.cu ---

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>       // For clock(), time(), srand()
#include <assert.h>     // For assert()
#include <cuda_runtime.h> // Main CUDA header
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.

// Forward declaration for correctness test if needed elsewhere
// (Not strictly necessary if main is commented out)
// void matrixMultiplyCorrectness();

// CPU vector Addition using C arrays and clock() for timing
long long cpuVectorAddition_int(int* A, int* B, size_t size) {
    clock_t start = clock();
    for(size_t i = 0; i < size; i++) {
        A[i] += B[i]; // Modify A in place
    }
    clock_t stop = clock();
    // Calculate duration in milliseconds
    long long duration_ms = (long long)(((double)(stop - start) / CLOCKS_PER_SEC) * 1000.0);
    printf("Speed of CPU vector Addition: %lld milliseconds\n", duration_ms);
    return duration_ms;
}

__global__ void add_kernel_int(int *a, int *b, int*c) {
    // Use blockIdx.x * blockDim.x + threadIdx.x for more general kernels
    // Here, assuming one thread per element directly mapped by blockIdx
    int idx = blockIdx.x;
    // Add boundary check if grid size might exceed data size
    // if (idx < N) { // Where N is the total size
       c[idx] = a[idx] + b[idx];
    // }
}

// GPU vector Addition using Pointers and CUDA Events
float gpuVectorAddition_int(const int* h_A, const int* h_B, size_t n) {

    int *d_a, *d_b, *d_c;
    // Allocate host memory for result (optional, could modify h_A or return d_c if needed elsewhere)
    int* h_C = (int *)malloc(sizeof(int)*n);
    assert(h_C != NULL);

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, sizeof(int)*n); assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_b, sizeof(int)*n); assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_c, sizeof(int)*n); assert(err == cudaSuccess);

    // Copy input data from Host to Device
    err = cudaMemcpy(d_a, h_A, sizeof(int)*n, cudaMemcpyHostToDevice); assert(err == cudaSuccess);
    err = cudaMemcpy(d_b, h_B, sizeof(int)*n, cudaMemcpyHostToDevice); assert(err == cudaSuccess);

    // Timing events
    cudaEvent_t launch_begin, launch_end;
    err = cudaEventCreate(&launch_begin); assert(err == cudaSuccess);
    err = cudaEventCreate(&launch_end);   assert(err == cudaSuccess);

    // Warmup run (optional but good practice)
    add_kernel_int<<<n,1>>>(d_a, d_b, d_c); // num blocks, num_threads per block
    err = cudaGetLastError(); assert(err == cudaSuccess); // Check for launch errors
    err = cudaDeviceSynchronize(); assert(err == cudaSuccess); // Wait for warmup completion

    float total_time = 0;
    int num_times = 10;
    // Get average of multiple runs
    for(int i = 0; i < num_times; i++) {
        err = cudaEventRecord(launch_begin, 0); assert(err == cudaSuccess);
        add_kernel_int<<<n,1>>>(d_a, d_b, d_c);
        err = cudaGetLastError(); assert(err == cudaSuccess); // Check launch error
        err = cudaEventRecord(launch_end, 0);   assert(err == cudaSuccess);
        err = cudaEventSynchronize(launch_end); assert(err == cudaSuccess); // Wait for kernel completion

        float time_ms = 0;
        err = cudaEventElapsedTime(&time_ms, launch_begin, launch_end); assert(err == cudaSuccess);
        total_time += time_ms;
    }

    // Destroy events
    err = cudaEventDestroy(launch_begin); assert(err == cudaSuccess);
    err = cudaEventDestroy(launch_end);   assert(err == cudaSuccess);

    total_time /= num_times;
    printf("Speed of GPU vector Addition: %.4f milliseconds\n", total_time);

    // Copy result memory back from Device to Host
    err = cudaMemcpy(h_C, d_c, sizeof(int)*n, cudaMemcpyDeviceToHost); assert(err == cudaSuccess);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host result memory
    free(h_C);

    return total_time; // Return time in milliseconds
}

void vectorAdditionSpeedTest() {
    size_t n = 1000000;
    // printf("Vector size: %zu\n", n);

    // Allocate and initialize host arrays
    int* A = (int*)malloc(n * sizeof(int));
    int* B = (int*)malloc(n * sizeof(int));
    int* A_copy = (int*)malloc(n * sizeof(int)); // Copy for CPU test as cpuVectorAddition modifies A
    assert(A != NULL && B != NULL && A_copy != NULL);

    for(size_t i = 0; i < n; i++) {
        A[i] = 1;
        B[i] = -1;
        A_copy[i] = A[i]; // Create copy before CPU modifies it
    }

    // Seed random number generator (if needed elsewhere, otherwise not used here)
    // srand(time(NULL));

    long long timeCpu_ms = cpuVectorAddition_int(A_copy, B, n); // Use copy for CPU test
    float timeGpu_ms = gpuVectorAddition_int(A, B, n);       // Use original for GPU test

    // Avoid division by zero if GPU time is extremely small or zero
    if (timeGpu_ms > 0.0f) {
        printf("Speedup over CPU for addition is: %.2f\n", (float)timeCpu_ms / timeGpu_ms);
    } else {
        printf("GPU execution too fast or failed, cannot calculate speedup.\n");
    }

    // Free host memory
    free(A);
    free(B);
    free(A_copy);
}

// Observation for CPU vs GPU compute in vector addition:
// 1. CUDA has a start-up overhead (context creation, data transfer).
// 2. Memory transfer (Host <-> Device) is relatively slow.
// 3. For simple, memory-bound operations like vector addition with insufficient data size
//    or computation intensity, the overheads can outweigh the parallel computation benefits.

// Matrix multiplication kernel (float version)
__global__ void mm_float(float* a, float* b, float* c, int width) {
    // Simple kernel: one block calculates one row of C, one thread calculates one element of C
    int row = blockIdx.x;    // Row index in C (and A)
    int col = threadIdx.x;   // Column index in C (and B)

    // Check boundaries (optional but safer)
    // if (row >= height || col >= width) return; // Assuming grid is height x width

    float temp = 0.0f;
    // width here is the common dimension (cols of A, rows of B)
    for(int k = 0; k < width; k++) {
        // A[row, k] = a[row * width + k]
        // B[k, col] = b[k * width + col] (assuming square matrices or width is cols of A)
        temp += a[row * width + k] * b[k * width + col];
    }
    // C[row, col] = c[row * width + col]
    c[row * width + col] = temp;
}

// GPU Matrix Multiplication using Pointers (float version)
float gpuMatrixMultiplication_float(const float* h_A, const float* h_B, int size, bool print) {

    size_t n_elements = (size_t)size * size;

    float *d_a, *d_b, *d_c;
    // Allocate host memory for result
    float* h_C = (float *)malloc(sizeof(float) * n_elements);
    assert(h_C != NULL);

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, sizeof(float) * n_elements); assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_b, sizeof(float) * n_elements); assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_c, sizeof(float) * n_elements); assert(err == cudaSuccess);

    // Copy input data from Host to Device
    err = cudaMemcpy(d_a, h_A, sizeof(float) * n_elements, cudaMemcpyHostToDevice); assert(err == cudaSuccess);
    err = cudaMemcpy(d_b, h_B, sizeof(float) * n_elements, cudaMemcpyHostToDevice); assert(err == cudaSuccess);

    // Timing events
    cudaEvent_t launch_begin, launch_end;
    err = cudaEventCreate(&launch_begin); assert(err == cudaSuccess);
    err = cudaEventCreate(&launch_end);   assert(err == cudaSuccess);

    // Kernel launch configuration: gridDim=(size), blockDim=(size)
    // Each block calculates one row, each thread in a block calculates one element in that row
    dim3 gridDim(size);     // Number of blocks = number of rows in C
    dim3 blockDim(size);    // Number of threads per block = number of columns in C
    int width = size;       // Common dimension for multiplication

    // Warmup run
    mm_float<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    err = cudaGetLastError(); assert(err == cudaSuccess);
    err = cudaDeviceSynchronize(); assert(err == cudaSuccess);

    float total_time = 0;
    int num_times = 10;
    if (!print) {
        // Get average of multiple runs
        for(int i = 0; i < num_times; i++) {
            err = cudaEventRecord(launch_begin, 0); assert(err == cudaSuccess);
            mm_float<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
            err = cudaGetLastError(); assert(err == cudaSuccess);
            err = cudaEventRecord(launch_end, 0);   assert(err == cudaSuccess);
            err = cudaEventSynchronize(launch_end); assert(err == cudaSuccess);

            float time_ms = 0;
            err = cudaEventElapsedTime(&time_ms, launch_begin, launch_end); assert(err == cudaSuccess);
            total_time += time_ms;
        }
        total_time /= num_times;
    } else {
        // Single run for correctness printing
        mm_float<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
        err = cudaGetLastError(); assert(err == cudaSuccess);
        err = cudaDeviceSynchronize(); assert(err == cudaSuccess);
    }


    // Destroy events
    err = cudaEventDestroy(launch_begin); assert(err == cudaSuccess);
    err = cudaEventDestroy(launch_end);   assert(err == cudaSuccess);

    // Copy result memory back from Device to Host
    err = cudaMemcpy(h_C, d_c, sizeof(float) * n_elements, cudaMemcpyDeviceToHost); assert(err == cudaSuccess);

    if(print) {
        printf("GPU Result:\n");
        for(size_t i = 0; i < n_elements; i++) {
            printf("%.2f ", h_C[i]);
            if ((i + 1) % size == 0) printf("\n");
        }
        printf("\n");
    } else {
        printf("Speed of GPU Matrix Multiplication: %.4f milliseconds\n", total_time);
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host result memory
    free(h_C);

    return total_time; // Return time in milliseconds
}

// CPU Matrix Multiplication (float version)
long long mmCpu_float(const float* a, const float* b, int n, bool print) {

    size_t n_elements = (size_t)n * n;
    // Allocate result matrix C, initialized to zero
    float* c = (float*)calloc(n_elements, sizeof(float));
    assert(c != NULL);

    clock_t start = clock();

    // Standard O(n^3) matrix multiplication
    for(int i = 0; i < n; i++) {       // Row of C (and A)
        for(int j = 0; j < n; j++) {   // Col of C (and B)
            for(int k = 0; k < n; k++) { // Common dimension
                // C[i, j] += A[i, k] * B[k, j]
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    clock_t stop = clock();
    long long duration_ms = (long long)(((double)(stop - start) / CLOCKS_PER_SEC) * 1000.0);

    if(print) {
        printf("CPU Result:\n");
        for(size_t i = 0; i < n_elements; i++) {
            printf("%.2f ", c[i]);
            if ((i + 1) % n == 0) printf("\n");
        }
        printf("\n");
    } else {
        printf("Speed of CPU Matrix Multiplication: %lld milliseconds\n", duration_ms);
    }

    // Free result memory
    free(c);

    return duration_ms; // Return time in milliseconds
}

void matrixMultiplySpeedTest() {
    int size = 100; // Adjust size as needed (e.g., 100, 256, 512)
    size_t n_elements = (size_t)size * size;

    // Allocate host matrices
    float* A = (float*)malloc(n_elements * sizeof(float));
    float* B = (float*)malloc(n_elements * sizeof(float));
    assert(A != NULL && B != NULL);

    // Initialize matrices (e.g., with simple values or random)
    srand(time(NULL)); // Seed random
    for(size_t i = 0; i < n_elements; i++) {
        A[i] = (float)(rand() % 10); // Simple random values 0-9
        B[i] = (float)(rand() % 10);
    }

    printf("--- Matrix Multiplication Speed Test (Size: %d x %d) ---\n", size, size);
    float gpuSpeed_ms = gpuMatrixMultiplication_float(A, B, size, false);
    long long cpuSpeed_ms = mmCpu_float(A, B, size, false);

    if (gpuSpeed_ms > 0.0f) {
        printf("Speedup of GPU over CPU is: %.2f\n", (float)cpuSpeed_ms / gpuSpeed_ms);
    } else {
         printf("GPU execution too fast or failed, cannot calculate speedup.\n");
    }
     printf("--------------------------------------------------------\n");

    // Free host memory
    free(A);
    free(B);
}


void matrixMultiplyCorrectness() {
    int size = 4;
    // Use float literals
    float A[] = {3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f};
    float B[] = {3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f,
                 3.0f, 1.0f, 2.0f, 4.0f};

    printf("--- Matrix Multiplication Correctness Test (Size: %d x %d) ---\n", size, size);
    gpuMatrixMultiplication_float(A, B, size, true); // Print GPU result
    mmCpu_float(A, B, size, true);         // Print CPU result
    printf("-----------------------------------------------------------\n");
}

// Example main (keep commented as requested)
/*
int main() {
    vectorAdditionSpeedTest();
    printf("\n");
    matrixMultiplySpeedTest();
    printf("\n");
    matrixMultiplyCorrectness();
    return 0; // Use 0 for successful exit in C
}
*/

// --- END OF FILE cudaOps.cu ---