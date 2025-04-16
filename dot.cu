// --- START OF FILE dot.cu ---

#include "buildTensorflow_float3.h" // Provides Matrix_float definition
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Include guard for the header part (declaration)
#ifndef __GPU_DOT_FLOAT_CU_INCLUDED__
#define __GPU_DOT_FLOAT_CU_INCLUDED__

// Forward declaration of the GPU dot product utility function
void dotGPU_float(float* res_data, const Matrix_float* lhs, const Matrix_float* rhs, int start, int startRes);

#endif // __GPU_DOT_FLOAT_CU_INCLUDED__


// --- Implementation ---

// Matrix multiplication kernel (float version)
// Duplicate from cudaOps.cu - necessary if not using a shared header
// Make it static if compiled/linked together to avoid symbol conflicts,
// but as separate .cu files, it's okay without static.
__global__ void mm_float_dot(float* a, float* b, float* c, int width) {
    // Simple kernel: one block calculates one row of C, one thread calculates one element of C
    int row = blockIdx.x;    // Row index in C (and A)
    int col = threadIdx.x;   // Column index in C (and B)

    // Boundary check recommended if grid/block dims might exceed actual matrix dims
    // Example: if (row >= gridDim.x || col >= blockDim.x) return;

    float temp = 0.0f;
    // 'width' here is the common dimension (cols of A, rows of B)
    for(int k = 0; k < width; k++) {
        // A[row, k] = a[row * width + k]
        // B[k, col] = b[k * width + col]
        temp += a[row * width + k] * b[k * width + col];
    }
    // C[row, col] = c[row * width + col]
    c[row * width + col] = temp;
}


/**
 * @brief Performs matrix multiplication C = A * B using CUDA for a specific slice of A.
 *
 * This function computes the dot product of a 2D slice of the potentially N-D `lhs` matrix
 * with the 2D `rhs` matrix. The slice of `lhs` starts at the linear index `start`.
 * The result is written into `res_data` starting at the linear index `startRes`.
 * Assumes the relevant dimensions for multiplication are the last two of `lhs` and the two of `rhs`.
 *
 * @param res_data Pointer to the host memory buffer where the result (C) will be stored.
 *                 Must be large enough to hold the result (row1 * col2 elements) starting at startRes.
 * @param lhs Pointer to the left-hand side matrix (A). Can be N-D.
 * @param rhs Pointer to the right-hand side matrix (B). Assumed to be 2D.
 * @param start The starting linear index within `lhs->val->val` for the 2D slice to be multiplied.
 * @param startRes The starting linear index within `res_data` where the result should be written.
 */
void dotGPU_float(float* res_data, const Matrix_float* lhs, const Matrix_float* rhs, int start, int startRes) {
    // Ensure matrices and their data/shapes are valid
    assert(lhs != NULL && lhs->val != NULL && lhs->shape != NULL && lhs->num_dims >= 2);
    assert(rhs != NULL && rhs->val != NULL && rhs->shape != NULL && rhs->num_dims == 2);
    assert(res_data != NULL);

    // Get dimensions for the actual 2D multiplication
    // Assumes the last two dimensions of lhs are involved
    int row1 = lhs->shape[lhs->num_dims - 2];
    int col1 = lhs->shape[lhs->num_dims - 1]; // Common dimension
    // Assumes rhs is 2D
    int row2 = rhs->shape[rhs->num_dims - 2]; // Common dimension
    int col2 = rhs->shape[rhs->num_dims - 1];

    // Sanity Check: Inner dimensions must match for matrix multiplication
    assert(col1 == row2);
    assert(row1 > 0 && col1 > 0 && row2 > 0 && col2 > 0); // Dimensions must be positive

    // Calculate sizes for memory allocation/copy
    size_t size_A_slice = (size_t)row1 * col1;
    size_t size_B = (size_t)row2 * col2;
    size_t size_C_slice = (size_t)row1 * col2;

    // Pointers to host data (use ->val->val from the C struct)
    const float* h_A_start = lhs->val->val + start; // Point to the start of the relevant slice in lhs
    const float* h_B = rhs->val->val;             // Point to the start of rhs data
    float* h_C_start = res_data + startRes;       // Point to the start of the result destination

    // Device pointers
    float *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, sizeof(float) * size_A_slice); assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_b, sizeof(float) * size_B);       assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_c, sizeof(float) * size_C_slice); assert(err == cudaSuccess);

    // Copy input matrices from Host to Device
    // Copy the specific slice of A (lhs)
    err = cudaMemcpy(d_a, h_A_start, sizeof(float) * size_A_slice, cudaMemcpyHostToDevice); assert(err == cudaSuccess);
    // Copy the entire B (rhs)
    err = cudaMemcpy(d_b, h_B, sizeof(float) * size_B, cudaMemcpyHostToDevice);             assert(err == cudaSuccess);

    // Define kernel launch parameters
    // Grid dimension corresponds to rows of the result (row1)
    // Block dimension corresponds to columns of the result (col2)
    dim3 gridDim(row1);
    dim3 blockDim(col2);
    int width = col1; // The common dimension passed to the kernel

    // Launch the matrix multiplication kernel
    mm_float_dot<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);

    // Check for kernel launch errors (optional but recommended)
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        assert(err == cudaSuccess); // Force stop on error
    }

    // Synchronize device (wait for kernel completion) before copying back result
    // Important, as kernel launch is asynchronous.
    err = cudaDeviceSynchronize();
     if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronize failed: %s\n", cudaGetErrorString(err));
        assert(err == cudaSuccess);
    }

    // Copy result back from Device to Host into the correct position in res_data
    err = cudaMemcpy(h_C_start, d_c, sizeof(float) * size_C_slice, cudaMemcpyDeviceToHost); assert(err == cudaSuccess);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// --- END OF FILE dot.cu ---