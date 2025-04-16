/*

Okay, I understand. The compiler errors you've provided clearly show the mismatch between how the functions are *called* in `main_c_tests.c` and how they are *defined* in `buildTensorflow_float2.h`.

**Crucially, the errors are NOT in the `buildTensorflow_float2.h` file itself. The header file defines the functions correctly, but the `main_c_tests.c` file has not been updated to use these functions according to their new definitions.**

Let's break down the errors and relate them to the header file:

1.  **`error: too few arguments to function ‘tensor_float_create_from_values’`**:
    *   **Header Definition (`buildTensorflow_float2.h`, line 854):**
        ```c
        Tensor_float* tensor_float_create_from_values(const float* val_data, const int* shape, int num_dims, bool requires_grad)
        ```
        This definition correctly requires **4 arguments**, including the `bool requires_grad` at the end.
    *   **Error Cause (`main_c_tests.c`):** The calls to `tensor_float_create_from_values` in your `main_c_tests.c` file are still providing only **3 arguments**. For example: `tensor_float_create_from_values(v1, s1, 1)` is missing the `requires_grad` argument.
    *   **Fix:** You MUST modify **`main_c_tests.c`** to add the fourth argument (e.g., `true` or `false`) to every call to `tensor_float_create_from_values`. Example: `tensor_float_create_from_values(v1, s1, 1, true)`

2.  **`error: too few arguments to function ‘dense_float_create’`**:
    *   **Header Definition (`buildTensorflow_float2.h`, line 2394):**
        ```c
        Dense_float* dense_float_create(int input_size, int output_size, activation_enum act, initialisation_enum weight_init, initialisation_enum bias_init)
        ```
        This definition correctly requires **5 arguments**, including the `initialisation_enum bias_init` at the end.
    *   **Error Cause (`main_c_tests.c`):** The calls to `dense_float_create` in your `main_c_tests.c` file are still providing only **4 arguments**. For example: `dense_float_create(2, 5, ACTIVATION_SIGMOID, INITIALISATION_GLOROT)` is missing the `bias_init` argument.
    *   **Fix:** You MUST modify **`main_c_tests.c`** to add the fifth argument (e.g., `INITIALISATION_ZEROS` or `INITIALISATION_GLOROT`) to every call to `dense_float_create`. Example: `dense_float_create(2, 5, ACTIVATION_SIGMOID, INITIALISATION_GLOROT, INITIALISATION_ZEROS)`

The header file `buildTensorflow_float2.h` you provided in the prompt *already contains* the correct definitions that lead to these errors when used with the old `main_c_tests.c` code. The changes I mentioned in the previous response (renaming `INITIALISATION_GLOROT_UNIFORM` back to `INITIALISATION_GLOROT` and uncommenting `SGD_Optimizer_float_minimise`) were already incorporated into the header file content you provided.

**Therefore, no further changes are needed in `buildTensorflow_float2.h`.** The file is correct as is.

**The solution lies entirely in updating the calls within `main_c_tests.c` to match the function signatures defined in `buildTensorflow_float2.h`.**

Here is the content of `buildTensorflow_float2.h` again for reference, confirming it has the correct signatures that require the changes in `main_c_tests.c`.

```c
--- START OF FILE buildTensorflow_float2.h ---
*/

/*
Okay, I will modify the `buildTensorflow_float1.h` header file to address the compiler errors based on the information provided.

The primary changes required in the header file itself are:

1.  **Rename `INITIALISATION_GLOROT_UNIFORM` back to `INITIALISATION_GLOROT`**: This will match the usage in the `main_c_tests.c` file.
2.  **Uncomment `SGD_Optimizer_float_minimise`**: This function is being called in the test file, so its definition needs to be available.

**Important Note:** The majority of the errors (`too few arguments to function ‘tensor_float_create_from_values’` and `too few arguments to function ‘dense_float_create’`) are *not* due to errors in the header file's *definition* of these functions, but rather because the *calls* to these functions in `main_c_tests.c` have not been updated to match the signatures defined in the header.

*   `tensor_float_create_from_values` requires a `bool requires_grad` argument.
*   `dense_float_create` requires both a `weight_init` and a `bias_init` argument.

**To fully resolve the errors, `main_c_tests.c` MUST be modified.** However, I will apply the necessary changes to the header file as requested.

--- START OF REVISED FILE buildTensorflow_float1.h ---
*/


/*
```c
Okay, here's the translation of the C++ header file `buildTensorflow (2).h` and its implicitly included components into a single C header file.

**Key C Emulation Strategies Used:**

1.  **Templates (`template<typename T>`):** Removed. The code is now specific to `float`. Structs and functions are renamed (e.g., `Tensor_float`, `matrix_float_add`). If other types were needed, separate versions would have to be created (e.g., `Tensor_double`).
2.  **Classes/Structs with Methods:** C++ classes become C `struct`s containing only data. Member functions become standalone C functions, taking a pointer to the struct instance as their first argument (conventionally named `self`).
3.  **Constructors/Destructors:** Replaced with explicit `_create` and `_destroy` functions for allocation/initialization and deallocation/cleanup.
4.  **Operator Overloading:** Replaced with named functions (e.g., `tensor_float_add` instead of `operator+`).
5.  **`std::vector`:** Emulated using dynamically allocated arrays (`float*`, `int*`) and explicit size/capacity management where needed. For simplicity in direct translation, often just using `malloc` based on calculated sizes and storing the pointer and size/dimensions.
6.  **Namespaces:** Removed. Functions previously in namespaces are prefixed (e.g., `utils_glorotInit_float`, `tensorOps_add_float`).
7.  **Inheritance (`Operation` hierarchy):** Emulated using struct composition and function pointers. The "derived" structs (e.g., `AddOperation_float`) contain the "base" struct (`Operation_float`) as their *first* member. The base struct contains function pointers (`backward`, `forward`) that are set by the `_create` function of the specific operation type. This allows polymorphic calls via the base struct pointer.
8.  **`std::queue`, `std::unordered_set`:** Emulated using simple dynamic arrays and manual management (linear search for set 'contains'). More robust implementations would require dedicated C data structure libraries or custom implementations.
9.  **Memory Management:** Uses `malloc` and `free`. **Crucially, the memory management in this translation mirrors the potential issues of the original C++ (like the Tensor destructor only freeing `backOp`). Robust C memory management would require a more careful design of ownership.**
10. **`std::iostream`, `std::cmath`, `std::random`:** Replaced with C equivalents (`stdio.h`, `math.h`, `stdlib.h`).
11. **Pass-by-reference:** Replaced with passing pointers.

```c
*/


/* Combined C Header File: buildTensorflow_float.h */
#ifndef BUILD_TENSORFLOW_FLOAT_H
#define BUILD_TENSORFLOW_FLOAT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // For memcpy
#include <assert.h>
#include <math.h>   // For exp, pow, sqrt
#include <time.h>   // For srand

// Define common macro to suppress unused parameter warnings portably
#define UNUSED(x) (void)(x)

// Forward Declarations (necessary in C for interdependent structs/functions)
// --- From types/matrix.h ---
typedef struct Matrix_float Matrix_float;

// --- From types/tensor.h ---
typedef struct Tensor_float Tensor_float;

// --- From operations/operation.h ---
typedef struct Operation_float Operation_float;

// --- From specific operation headers ---
typedef struct AddOperation_float AddOperation_float;
typedef struct MultiplyOperation_float MultiplyOperation_float;
typedef struct DivideOperation_float DivideOperation_float;
typedef struct ExponentOperation_float ExponentOperation_float;
typedef struct DotOperation_float DotOperation_float;
typedef struct SigmoidOperation_float SigmoidOperation_float;
typedef struct PowerOperation_float PowerOperation_float;

// --- From optims/optim.h ---
typedef struct Optimizer_float Optimizer_float;

// --- From optims/sgd.h ---
typedef struct SGD_Optimizer_float SGD_Optimizer_float;

// --- From data/dataloader.h ---
// Define pair structure for data loader
typedef struct DataLoader_Pair_float_float 
{
    float input;
    float target;
} DataLoader_Pair_float_float;

typedef struct DataLoader_float_float DataLoader_float_float;

// --- From data/celsius2fahrenheit.h ---
typedef struct Celsius2Fahrenheit_DataLoader_float_float Celsius2Fahrenheit_DataLoader_float_float;

// --- From dense.h ---
typedef struct Dense_float Dense_float;


// ========================================
// START OF "overloads/vector.h" content (as functions)
// ========================================
/*
    Vector Operations (Emulated). This file over loads several operator for vector like elementwise

    1. Addition
    2. Multiplication
    3. Division
    4. Exponent
    5. Power

    with scalars as well as vectors (scalar vector combo)
*/

// Helper: Allocates result vector, caller must free.
float* vector_float_allocate(size_t size) 
{
    float* result = (float*)malloc(size * sizeof(float));
    assert(result != NULL && "Memory allocation failed for vector operation result");
    return result;
}

// Multiplication (vector * vector)
float* vector_float_multiply_vv(const float* a, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = a[i] * b[i];
    }
    return arr;
}

// Scalar Multiplication (scalar * vector)
float* vector_float_multiply_sv(float scalar, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = scalar * b[i];
    }
    return arr;
}

// Addition (vector + vector)
float* vector_float_add_vv(const float* a, const float* b, size_t size) {
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) {
        arr[i] = a[i] + b[i];
    }
    return arr;
}

// Scalar Addition (scalar + vector)
float* vector_float_add_sv(float scalar, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = scalar + b[i];
    }
    return arr;
}

// Subtraction (vector - vector)
float* vector_float_subtract_vv(const float* a, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = a[i] - b[i];
    }
    return arr;
}

// Scalar Subtraction (scalar - vector)
float* vector_float_subtract_sv(float scalar, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = scalar - b[i];
    }
    return arr;
}

// Vector Divide (vector / vector)
float* vector_float_divide_vv(const float* a, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        // Consider adding check for division by zero
        assert(b[i] != 0.0f && "Division by zero error in vector_float_divide_vv");
        arr[i] = a[i] / b[i];
    }
    return arr;
}

// Scalar divide (scalar / vector)
float* vector_float_divide_sv(float scalar, const float* b, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        // Consider adding check for division by zero
        assert(b[i] != 0.0f && "Division by zero error in vector_float_divide_sv");
        arr[i] = scalar / b[i];
    }
    return arr;
}

// Power Operation (vector ^ scalar)
float* vector_float_power_vs(const float* a, float scalar_pow, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = powf(a[i], scalar_pow); // Use powf for float
    }
    return arr;
}

// Exponent Operation (exp(vector))
float* vector_float_exponent(const float* a, size_t size) 
{
    float* arr = vector_float_allocate(size);
    for (size_t i = 0; i < size; i++) 
    {
        arr[i] = expf(a[i]); // Use expf for float
    }
    return arr;
}

// isEquals operator (vector == vector)
bool vector_float_equals(const float* a, const int* shape_a, int dims_a,
                         const float* b, const int* shape_b, int dims_b) 
{
    if (dims_a != dims_b) return false;
    size_t n_a = 1;
    size_t n_b = 1;
    for(int i = 0; i < dims_a; ++i) 
    {
        if (shape_a[i] != shape_b[i]) return false;
        n_a *= shape_a[i];
        n_b *= shape_b[i];
    }
    if (n_a != n_b) return false; // Should be guaranteed by shape check, but good practice

    for (size_t i = 0; i < n_a; i++) 
    {
        // Use epsilon comparison for floats if exact match is not required
        if (fabsf(a[i] - b[i]) > 1e-6f) 
        {
            return false;
        }
    }
    return true;
}

// ========================================
// END OF "overloads/vector.h" content
// ========================================


// ========================================
// START OF "types/matrix.h" content
// ========================================
/*
    This file defines the Matrix struct. Matrix supports creation of n dimensional arrays
    It supports various arithmetic and matrix operations.
*/

struct Matrix_float 
{
    float* val; // underlying data structure that holds the values of matrix
    int* shape; // shape of that matrix
    int num_dims; // Number of dimensions
    size_t num_elements; // Total number of elements (product of shape)

    /*
        This array tracks for each dimension- the total number of elements
        left to encounter if I continue deeper into the remaining dimensions.
        Size is num_dims. Also known as strides in row-major order.
    */
    size_t* elemsEncounteredPerDim; // Changed to size_t to match num_elements

    // Check whether GPU is accessible or not (Placeholder - not functional in C translation)
    bool gpu;
};

// --- Private Helper Function Prototypes ---
bool matrix_float_verifyShape(size_t num_elements_val, const int* shape, int num_dims); // Removed unused 'val' pointer
size_t* matrix_float_computeShapes(const int* shapes, int num_dims); // Changed return type to size_t*
bool matrix_float_verifyShapeForElementwiseOperation(const int* shape1, int dims1, const int* shape2, int dims2);
bool matrix_float_verifyShapeForDotProductOperation(const int* shape1, int dims1, const int* shape2, int dims2);
void matrix_float_matmulUtil(Matrix_float* self, float* res, const Matrix_float* rhs, size_t start, size_t startRes); // Changed indices to size_t
void matrix_float_matmulRecursive(Matrix_float* self, float* res, const Matrix_float* rhs, int* stack, const size_t* resElems, int dim); // Changed resElems to size_t*
// void matrix_float_printRecursive(Matrix_float* self, FILE* out, int* stack, int dim); // Adjusted print helper

// --- Public Function Prototypes ---
Matrix_float* matrix_float_create(const float* val, const int* shape, int num_dims);
void matrix_float_destroy(Matrix_float* matrix);
void matrix_float_print(Matrix_float* self, FILE* out); // Replaces operator<<

// --- Operation Functions (replacing operator overloads) ---
Matrix_float* matrix_float_add(const Matrix_float* lhs, const Matrix_float* rhs);
Matrix_float* matrix_float_subtract(const Matrix_float* lhs, const Matrix_float* rhs);
Matrix_float* matrix_float_divide_elementwise(const Matrix_float* lhs, const Matrix_float* rhs);
Matrix_float* matrix_float_multiply_elementwise(const Matrix_float* lhs, const Matrix_float* rhs);
Matrix_float* matrix_float_power_scalar(const Matrix_float* lhs, float exponent);
Matrix_float* matrix_float_exp(const Matrix_float* lhs);
Matrix_float* matrix_float_dot(const Matrix_float* lhs, const Matrix_float* rhs);

// --- Scalar Operation Functions ---
Matrix_float* matrix_float_add_scalar(float scalar, const Matrix_float* rhs);
Matrix_float* matrix_float_subtract_scalar_lhs(float scalar, const Matrix_float* rhs); // scalar - matrix
Matrix_float* matrix_float_divide_scalar_lhs(float scalar, const Matrix_float* rhs);   // scalar / matrix
Matrix_float* matrix_float_multiply_scalar(float scalar, const Matrix_float* rhs);

// --- Function Implementations --- (Ideally in a .c file, but included here for single-header request)

// Verifies that the shape provided and val vector provided are compatible in size
bool matrix_float_verifyShape(size_t num_elements_val, const int* shape, int num_dims) 
{
    // UNUSED(val); // Mark val as unused - Removed parameter instead
    size_t p = 1;
    for (int i = 0; i < num_dims; i++) 
    {
        // Add check for non-positive dimension size
        assert(shape[i] > 0 && "Matrix dimension shape must be positive");
        p *= shape[i];
    }
    return (p == num_elements_val);
}

/*
    This function is used to compute the elemsEncounteredPerDim variable (strides).
    See it's description for more detail
*/
size_t* matrix_float_computeShapes(const int* shapes, int num_dims) 
{
    if (num_dims == 0) return NULL;
    size_t* elems = (size_t*)malloc(num_dims * sizeof(size_t));
    assert(elems != NULL);
    size_t p = 1;
    for (int i = num_dims - 1; i >= 0; i--) 
    {
        elems[i] = p;
        assert(shapes[i] > 0 && "Matrix dimension shape must be positive");
        // Check for potential overflow if dimensions are huge
        // size_t next_p = p * shapes[i];
        // assert(next_p / shapes[i] == p && "Overflow calculating total elements");
        p *= shapes[i];
    }
    return elems;
}

/*
    This function checks that the 2 matrices are compatible with each other to perform elementwise
    operations. Operations such as *, + and / are elementwise operations and they require the two
    matrices to have the same shape.
*/
bool matrix_float_verifyShapeForElementwiseOperation(const int* shape1, int dims1, const int* shape2, int dims2) {
    if (dims1 != dims2) 
    {
        return false;
    }
    for (int i = 0; i < dims1; i++) 
    {
        if (shape1[i] != shape2[i]) 
        {
            return false;
        }
    }
    return true;
}

/*
    This function verifies that the 2 matrices are compatible for dot product multiplication.
    Currently, this dot product functionality allows the lhs matrix (with shape1) to have 2 or
    more dimensions. The rhs matrix should have 2 dimensions. The multiplication will occur by
    traversing through the n-2 dimensions of matrix 1, until we get a 2d matrix of lhs. Then we
    perform matrix multiplication with this 2 d lhs matrix and the rhs matrix. This function
    performs a check to verify whether the two matrices are compatible for matrix multiplication.
*/
bool matrix_float_verifyShapeForDotProductOperation(const int* shape1, int dims1, const int* shape2, int dims2) {
    if (dims1 < 2 || dims2 != 2) 
    {
         fprintf(stderr, "Dot product requires lhs dims >= 2 and rhs dims == 2. Got %d and %d.\n", dims1, dims2);
        return false;
    }
    int col1 = shape1[dims1 - 1];
    int row2 = shape2[dims2 - 2]; // Equivalent to shape2[0] since dims2 is 2

    if (col1 != row2) 
    {
        fprintf(stderr, "Dot product shape mismatch: lhs last dim (%d) != rhs first dim (%d).\n", col1, row2);
        return false;
    }
    return true;
}

/*
    This is a utility function for matrix multiplication that performs the actual dot product.
    It gets as input the reference of the result vector, the 2d rhs matrix, the start index that
    tracks the position from which to start in the n dimensional lhs matrix and resStart tracks
    the position to start in the result vector
*/
void matrix_float_matmulUtil(Matrix_float* self, float* res, const Matrix_float* rhs, size_t start, size_t startRes) 
{
    int row1 = self->shape[self->num_dims - 2];
    int col1 = self->shape[self->num_dims - 1];
    // int row2 = rhs->shape[rhs->num_dims - 2]; // == col1
    int col2 = rhs->shape[rhs->num_dims - 1];
    // Sanity Check (already done by verifyShapeForDotProductOperation, but good for util)
    // assert(col1 == row2);

    // O(row1 * col2 * col1) complexity
    for (int i = 0; i < row1; i++) 
    { // Iterate rows of result (and lhs slice)
        for (int k = 0; k < col2; k++) { // Iterate columns of result (and rhs)
            float sum = 0.0f;
            for (int j = 0; j < col1; j++) 
            { // Iterate columns of lhs slice / rows of rhs
                // Indexing: lhs[i, j] = self->val[start + i * col1 + j]
                // Indexing: rhs[j, k] = rhs->val[j * col2 + k] (since rhs is 2D)
                sum += self->val[start + (size_t)i * col1 + j] * rhs->val[(size_t)j * col2 + k];
            }
             // Indexing: res[i, k] = res[startRes + i * col2 + k]
            res[startRes + (size_t)i * col2 + k] = sum;
        }
    }
}


/*
    Recursive helper for n-dimensional matmul. See C++ comments for details.
    Uses a stack (passed as allocated array) to track indices.
*/
void matrix_float_matmulRecursive(Matrix_float* self, float* res, const Matrix_float* rhs, int* stack, const size_t* resElems, int dim) 
{
    // Case when we are down to the last 2 dimensions of the LHS matrix
    if (dim == self->num_dims - 2) 
    {
        size_t lhs_start_offset = 0;
        size_t res_start_offset = 0;
        for (int i = 0; i < dim; i++) 
        { // Iterate up to current dim
            lhs_start_offset += self->elemsEncounteredPerDim[i] * stack[i];
            res_start_offset += resElems[i] * stack[i]; // resElems are strides for result matrix
        }
        // Do normal 2D matrix multiplication now
        matrix_float_matmulUtil(self, res, rhs, lhs_start_offset, res_start_offset);
        return;
    }

    // Case when larger matrix is bigger than rhs matrix: recurse deeper
    for (int i = 0; i < self->shape[dim]; i++) 
    {
        stack[dim] = i; // Store current index for this dimension
        matrix_float_matmulRecursive(self, res, rhs, stack, resElems, dim + 1);
        // No pop needed as we overwrite stack[dim] in the loop
    }
}

// Recursive helper for printing
void matrix_float_printRecursive(Matrix_float* self, FILE* out, int* current_indices, size_t current_offset, int dim) 
{
    fprintf(out, "[ ");
    if (dim == self->num_dims - 1) 
    {
        // Last dimension, print elements
        for (int i = 0; i < self->shape[dim]; ++i) 
        {
            fprintf(out, "%.4f ", self->val[current_offset + i]);
        }
    } else 
    {
        // Not the last dimension, recurse
        for (int i = 0; i < self->shape[dim]; ++i) 
        {
            current_indices[dim] = i;
            size_t next_offset = current_offset + (size_t)i * self->elemsEncounteredPerDim[dim];
            matrix_float_printRecursive(self, out, current_indices, next_offset, dim + 1);
            if (i < self->shape[dim] - 1) {
                 // Add newline/spacing for better readability between sub-matrices
                 fprintf(out, "\n"); // Example: newline between rows in 2D
                  // Indentation based on 'dim'
                 for(int d=0; d<=dim; ++d) fprintf(out, "  ");
            }
        }
    }
    fprintf(out, "]");
    // Add newline after completing a dimension block if not the outermost call
    // and not the last element of the parent dimension.
    // This logic needs refinement for consistent ND printing layout.
    // if (dim > 0 && current_indices[dim-1] < self->shape[dim-1]-1) 
    // {
    //     fprintf(out, "\n");
    //     // Indentation could be added here based on 'dim'
    //     for(int d=0; d<dim; ++d) fprintf(out, "  ");
    // }

}

// Public print function (replaces operator<<)
void matrix_float_print(Matrix_float* self, FILE* out) 
{
    if (!self || !self->val || !self->shape || self->num_dims <= 0) 
    {
        fprintf(out, "[ Uninitialized or Empty Matrix ]\n");
        return;
    }
    fprintf(out, "Matrix (Shape: [");
    for(int i=0; i<self->num_dims; ++i) 
    {
        fprintf(out, "%d%s", self->shape[i], (i == self->num_dims - 1) ? "" : ", ");
    }
    fprintf(out, "], %zu elements, Strides: [", self->num_elements);
     for(int i=0; i<self->num_dims; ++i) 
     {
        fprintf(out, "%zu%s", self->elemsEncounteredPerDim[i], (i == self->num_dims - 1) ? "" : ", ");
     }
    fprintf(out, "]):\n");


    int* current_indices = (int*)calloc(self->num_dims, sizeof(int)); // Track position
    assert(current_indices != NULL);
    matrix_float_printRecursive(self, out, current_indices, 0, 0);
    fprintf(out, "\n");
    free(current_indices);
}


// Constructor for matrix, validates shape before creating object
Matrix_float* matrix_float_create(const float* val_data, const int* shape, int num_dims) 
{
    size_t num_elements = 1;
    for (int i = 0; i < num_dims; i++) 
    {
        assert(shape[i] > 0 && "Matrix dimension shape must be positive");
        num_elements *= (size_t)shape[i]; // Use size_t for calculation
    }
    // Verify size (using num_elements directly)
    // assert("Shape and number of elements are incompatible!" && matrix_float_verifyShape(num_elements, shape, num_dims)); // Assuming size is implicitly correct from calculation

    Matrix_float* matrix = (Matrix_float*)malloc(sizeof(Matrix_float));
    assert(matrix != NULL);

    matrix->num_dims = num_dims;
    matrix->num_elements = num_elements;
    matrix->gpu = false; // Default to false

    // Deep copy shape
    matrix->shape = (int*)malloc(num_dims * sizeof(int));
    assert(matrix->shape != NULL);
    memcpy(matrix->shape, shape, num_dims * sizeof(int));

    // Deep copy values
    matrix->val = (float*)malloc(num_elements * sizeof(float));
    assert(matrix->val != NULL);
    memcpy(matrix->val, val_data, num_elements * sizeof(float));

    // Compute strides
    matrix->elemsEncounteredPerDim = matrix_float_computeShapes(matrix->shape, matrix->num_dims);

    return matrix;
}

// Destructor for matrix
void matrix_float_destroy(Matrix_float* matrix) 
{
    if (matrix) 
    {
        free(matrix->val);
        free(matrix->shape);
        free(matrix->elemsEncounteredPerDim);
        free(matrix);
    }
}

// Performs elementwise addition
Matrix_float* matrix_float_add(const Matrix_float* lhs, const Matrix_float* rhs) 
{
    assert(lhs && rhs && "Input matrices cannot be NULL");
    assert(matrix_float_verifyShapeForElementwiseOperation(lhs->shape, lhs->num_dims, rhs->shape, rhs->num_dims) &&
           "Shapes aren't compatible for addition !");

    float* res_val = vector_float_add_vv(lhs->val, rhs->val, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val); // Create copies the data, so we can free the intermediate
    return result;
}

// Performs elementwise subtraction
Matrix_float* matrix_float_subtract(const Matrix_float* lhs, const Matrix_float* rhs) 
{
     assert(lhs && rhs && "Input matrices cannot be NULL");
     assert(matrix_float_verifyShapeForElementwiseOperation(lhs->shape, lhs->num_dims, rhs->shape, rhs->num_dims) &&
            "Shapes aren't compatible for subtraction !");

    float* res_val = vector_float_subtract_vv(lhs->val, rhs->val, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val);
    return result;
}

// Performs elementwise division
Matrix_float* matrix_float_divide_elementwise(const Matrix_float* lhs, const Matrix_float* rhs) 
{
     assert(lhs && rhs && "Input matrices cannot be NULL");
     assert(matrix_float_verifyShapeForElementwiseOperation(lhs->shape, lhs->num_dims, rhs->shape, rhs->num_dims) &&
            "Shapes aren't compatible for elementwise division !");
    // vector_float_divide_vv already checks for division by zero

    float* res_val = vector_float_divide_vv(lhs->val, rhs->val, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val);
    return result;
}

// Performs elementwise multiplication
Matrix_float* matrix_float_multiply_elementwise(const Matrix_float* lhs, const Matrix_float* rhs) 
{
     assert(lhs && rhs && "Input matrices cannot be NULL");
     assert(matrix_float_verifyShapeForElementwiseOperation(lhs->shape, lhs->num_dims, rhs->shape, rhs->num_dims) &&
            "Shapes aren't compatible for elementwise multiplication !");

    float* res_val = vector_float_multiply_vv(lhs->val, rhs->val, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val);
    return result;
}


// Performs power operation with scalar
Matrix_float* matrix_float_power_scalar(const Matrix_float* lhs, float exponent) 
{
    assert(lhs && "Input matrix cannot be NULL");
    float* res_val = vector_float_power_vs(lhs->val, exponent, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val);
    return result;
}

// Perfroms exponent operation on each value of matrix
Matrix_float* matrix_float_exp(const Matrix_float* lhs) 
{
    assert(lhs && "Input matrix cannot be NULL");
    float* res_val = vector_float_exponent(lhs->val, lhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, lhs->shape, lhs->num_dims);
    free(res_val);
    return result;
}

// Dot Product between 2 matrices. Check matmul for indepth description.
Matrix_float* matrix_float_dot(const Matrix_float* lhs, const Matrix_float* rhs) 
{
    assert(lhs && rhs && "Input matrices cannot be NULL");
    assert(matrix_float_verifyShapeForDotProductOperation(lhs->shape, lhs->num_dims, rhs->shape, rhs->num_dims) &&
           "Shapes aren't compatible for dot product !");

    // Calculate shape of result array
    int res_num_dims = lhs->num_dims; // Result has same number of dims as lhs
    int* res_shape = (int*)malloc(res_num_dims * sizeof(int));
    assert(res_shape != NULL);

    size_t res_num_elements = 1;
    // Copy leading dims from lhs
    for (int i = 0; i < lhs->num_dims - 2; i++) 
    {
         res_shape[i] = lhs->shape[i];
         res_num_elements *= (size_t)lhs->shape[i];
    }
    // Last two dimensions are rows_lhs x cols_rhs
    res_shape[res_num_dims - 2] = lhs->shape[lhs->num_dims - 2]; // rows from lhs
    res_shape[res_num_dims - 1] = rhs->shape[rhs->num_dims - 1]; // cols from rhs (rhs->shape[1])
    res_num_elements *= (size_t)res_shape[res_num_dims - 2] * res_shape[res_num_dims - 1];


    float* res_val = (float*)calloc(res_num_elements, sizeof(float)); // Use calloc for zero initialization
    assert(res_val != NULL);

    int stack_size = lhs->num_dims > 2 ? lhs->num_dims - 2 : 0;
    int* stack = NULL;
    if (stack_size > 0) 
    {
        stack = (int*)malloc(stack_size * sizeof(int)); // Stack for recursive calls
        assert(stack != NULL);
    }

    size_t* resElems = matrix_float_computeShapes(res_shape, res_num_dims); // Calculate strides for result
    assert(resElems != NULL || res_num_dims == 0);

    // Cast needed because the recursive function expects non-const self (though it doesn't modify it)
    matrix_float_matmulRecursive((Matrix_float*)lhs, res_val, rhs, stack, resElems, 0);

    Matrix_float* result = matrix_float_create(res_val, res_shape, res_num_dims);

    // Cleanup
    free(res_val);
    free(res_shape);
    free(stack); // free(NULL) is safe
    free(resElems);

    return result;
}

// Divison with a scalar as divident (scalar / matrix)
Matrix_float* matrix_float_divide_scalar_lhs(float scalar, const Matrix_float* rhs) 
{
    assert(rhs && "Input matrix cannot be NULL");
    // vector_float_divide_sv already checks for division by zero
    float* res_val = vector_float_divide_sv(scalar, rhs->val, rhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, rhs->shape, rhs->num_dims);
    free(res_val);
    return result;
}

// Multiplication with a scalar (scalar * matrix)
Matrix_float* matrix_float_multiply_scalar(float scalar, const Matrix_float* rhs) 
{
    assert(rhs && "Input matrix cannot be NULL");
    float* res_val = vector_float_multiply_sv(scalar, rhs->val, rhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, rhs->shape, rhs->num_dims);
    free(res_val);
    return result;
}

// Addition with a scalar (scalar + matrix)
Matrix_float* matrix_float_add_scalar(float scalar, const Matrix_float* rhs) 
{
    assert(rhs && "Input matrix cannot be NULL");
    float* res_val = vector_float_add_sv(scalar, rhs->val, rhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, rhs->shape, rhs->num_dims);
    free(res_val);
    return result;
}

// Subtraction with a scalar (scalar - matrix)
Matrix_float* matrix_float_subtract_scalar_lhs(float scalar, const Matrix_float* rhs) 
{
    assert(rhs && "Input matrix cannot be NULL");
    float* res_val = vector_float_subtract_sv(scalar, rhs->val, rhs->num_elements);
    Matrix_float* result = matrix_float_create(res_val, rhs->shape, rhs->num_dims);
    free(res_val);
    return result;
}

// Sigmoid Operation (defined later in matrixOps section, declared here)
Matrix_float* matrixOps_sigmoid_float(const Matrix_float* a);

// Power Operation (defined later in matrixOps section, declared here)
Matrix_float* matrixOps_power_float(Matrix_float* a, float pow_val);

// ========================================
// END OF "types/matrix.h" content
// ========================================


// ========================================
// START OF "operations/operation.h" content
// ========================================
/*
    This file defines the Operation base struct which represents an operation
    performed on one or more tensors.
*/
struct Operation_float 
{
    Tensor_float *t1; // First operand (if applicable)
    Tensor_float *t2; // Second operand (if applicable)
    Tensor_float *t3; // Output tensor (result of forward pass)

    // Function pointers for polymorphism (emulating virtual functions)
    // 'self' pointer MUST be castable to the specific operation type (e.g., AddOperation_float*)
    void (*backward)(Operation_float* self, Matrix_float* grad);
    Tensor_float* (*forward)(Operation_float* self);
    void (*destroy)(Operation_float* self); // Virtual destructor emulation
};

// Base destroy function (can be NULL if nothing to free in base)
// Specific destroy functions should call this if needed, after freeing their own resources.
// In this case, the base has no allocated memory, only pointers managed elsewhere.
// However, derived ops might need to free t3 if they own it.
void operation_float_base_destroy(Operation_float* self) 
{
    // Base does nothing, but provides the function pointer slot.
    // If t3 ownership was defined here, it would be freed here.
    if (self) 
    {
         // Example if t3 was owned by the operation:
         // tensor_float_destroy(self->t3); // Requires tensor_float_destroy to be defined
         // NOTE: Current design has tensor_create_from_op make the tensor own the op,
         // and tensor_destroy tries to destroy the op. This creates circular dependency.
         // A better design: Graph owns ops and tensors, or ops own result tensors,
         // but tensors DO NOT own ops.
         // Let's assume ops own their result tensor t3.
         // This function will be called by derived destroyers.
    }
}

// ========================================
// END OF "operations/operation.h" content
// ========================================


// ========================================
// START OF "types/tensor.h" content
// ========================================
/*
    This file defines the Tensor struct. The Tensor is the base object used for creating
    neural networks. Each neuron in the network will be a tensor. In this project we will
    use matrices to represent layers of a neural network, hence each tensor would simply be
    a n dimensional matrix.

    Apart from this the tensor is aware of what operations have been performed with it. This
    is useful when we want to calculate the gradients of the tensor with respect to some
    other variable (a loss function generally with neural networks).
    To calculate these gradients, the tensor keeps a track of what operations have been
    performed with it as a operand.

    For now we are assuming that the tensor just goes through one operation, hence we have
    a frontOp denoting the operation that it is involved in and backOp is the operation that
    created this tensor.

    To know more about Operations head to the operation module and check out the operation.h
    file.
*/

struct Tensor_float 
{
    /*
        Value of the Tensor: Matrix struct stores n dimensional arrays.
    */
    Matrix_float* val; // Pointer to Matrix

    /*
        Gradient value of the tensor, also stored as a Matrix.
        Initialized to zero with the same shape as val. Can be NULL initially.
    */
    Matrix_float* grad; // Pointer to Matrix

    /*
        The Operations that the tensor is related with. The frontOp denotes the operation
        that the tensor is an operand to. The backOp operation is the operation which led
        to the creation of this tensor.

        Ownership Note (Revised):
        - `backOp`: A pointer to the operation that created this tensor. The Tensor DOES NOT own this Op.
                    The Op likely owns the Tensor (its result).
        - `frontOp`: A pointer to the operation where this tensor is used as an input. The Tensor DOES NOT own this Op.
                    The Op points back to this Tensor via its `t1` or `t2`.
        - This avoids circular ownership and double frees. Graph cleanup needs a separate mechanism.
    */
    Operation_float *frontOp; // Operation using this tensor as input
    Operation_float *backOp;  // Operation that produced this tensor
    bool requires_grad;       // Flag to indicate if gradients should be computed for this tensor
};

// --- Tensor Function Prototypes ---
Tensor_float* tensor_float_create_from_matrix(Matrix_float* val_matrix, bool requires_grad); // Takes ownership of val_matrix
Tensor_float* tensor_float_create_from_values(const float* val_data, const int* shape, int num_dims, bool requires_grad);
Tensor_float* tensor_float_create_from_op(Matrix_float* val_matrix, Operation_float* op, bool requires_grad); // Takes ownership of val_matrix
Tensor_float* tensor_float_copy(const Tensor_float* other); // Deep copy
void tensor_float_destroy(Tensor_float* tensor); // Destructor
void tensor_float_zeroGrad(Tensor_float* self);
void tensor_float_backward(Tensor_float* self, Matrix_float* grad_incoming); // With incoming gradient
void tensor_float_backward_default(Tensor_float* self); // Starts chain with gradient of 1s

// --- Operations returning Tensor (Emulating Operator Overloads/Methods) ---
// These are implemented via tensorOps_ functions later

// --- Function Implementations ---

/*
    This function is called during the initilaisation of Tensor. It sets the value of it's gradients to zero if needed.
    If grad matrix doesn't exist, it creates it.
*/
void tensor_float_zeroGrad(Tensor_float* self) 
{
    assert(self && self->val && "Tensor or its value matrix cannot be uninitialised during zeroGrad");
    assert(self->val->shape != NULL && self->val->num_dims > 0 && "The value matrix cannot be uninitialised during initialising zeros in tensor's gradient");

    if (!self->requires_grad) 
    {
        // If gradients are not required, ensure grad is NULL and return.
        matrix_float_destroy(self->grad); // Free if it exists
        self->grad = NULL;
        return;
    }

    if (self->grad) 
    {
        // If grad exists, just zero its values
        assert(matrix_float_verifyShapeForElementwiseOperation(self->val->shape, self->val->num_dims, self->grad->shape, self->grad->num_dims) && "Value and Gradient shapes mismatch in zeroGrad");
        memset(self->grad->val, 0, self->grad->num_elements * sizeof(float));
    } else 
    {
        // If grad doesn't exist, create it filled with zeros
        float* zero_data = (float*)calloc(self->val->num_elements, sizeof(float));
        assert(zero_data != NULL && "Memory allocation failed for zero gradient data");
        self->grad = matrix_float_create(zero_data, self->val->shape, self->val->num_dims);
        free(zero_data); // create copies data
    }
}

// Constructor to create Tensor from a Matrix (Takes ownership of val_matrix)
Tensor_float* tensor_float_create_from_matrix(Matrix_float* val_matrix, bool requires_grad) 
{
    assert(val_matrix != NULL);
    Tensor_float* tensor = (Tensor_float*)malloc(sizeof(Tensor_float));
    assert(tensor != NULL);
    tensor->val = val_matrix; // Takes ownership
    tensor->grad = NULL;      // Initialized lazily by zeroGrad if needed
    tensor->frontOp = NULL;
    tensor->backOp = NULL;
    tensor->requires_grad = requires_grad;
    if (requires_grad) 
    {
        tensor_float_zeroGrad(tensor); // Initialize grad matrix if needed
    }
    return tensor;
}

/*
    Constructor to create Tensor from a value array and shape vector.
    NOTE: This function requires the 'requires_grad' argument. Calls to this function
          in older code (like main_c_tests.c) must be updated to provide this boolean argument.
*/
Tensor_float* tensor_float_create_from_values(const float* val_data, const int* shape, int num_dims, bool requires_grad) 
{
    Matrix_float* val_matrix = matrix_float_create(val_data, shape, num_dims);
    // create_from_matrix takes ownership of val_matrix
    return tensor_float_create_from_matrix(val_matrix, requires_grad);
}


/*
    Constructor used by an Operation to create its result tensor.
    The operation (`op`) is set as the `backOp`. (Takes ownership of val_matrix)
*/
Tensor_float* tensor_float_create_from_op(Matrix_float* val_matrix, Operation_float* op, bool requires_grad) 
{
    assert(val_matrix != NULL);
    Tensor_float* tensor = (Tensor_float*)malloc(sizeof(Tensor_float));
    assert(tensor != NULL);
    tensor->val = val_matrix; // Takes ownership
    tensor->grad = NULL;
    tensor->frontOp = NULL;
    tensor->backOp = op; // Store pointer to creating operation
    tensor->requires_grad = requires_grad; // Grad requirement propagates or is set by op
    if (requires_grad) 
    {
        tensor_float_zeroGrad(tensor);
    }
    return tensor;
}

/*
    Copy constructor. Performs a deep copy of matrices. Ops pointers are shallow copied.
    Requires_grad flag is copied.
*/
Tensor_float* tensor_float_copy(const Tensor_float* other) 
{
    assert(other != NULL);
    Tensor_float* tensor = (Tensor_float*)malloc(sizeof(Tensor_float));
    assert(tensor != NULL);

    // Deep copy value matrix
    if (other->val) 
    {
        tensor->val = matrix_float_create(other->val->val, other->val->shape, other->val->num_dims);
    } else 
    {
        tensor->val = NULL;
    }

    // Deep copy gradient matrix (only if it exists)
    if (other->grad) 
    {
        tensor->grad = matrix_float_create(other->grad->val, other->grad->shape, other->grad->num_dims);
    } else 
    {
        tensor->grad = NULL;
    }

    // Shallow copy operation pointers
    tensor->frontOp = other->frontOp;
    tensor->backOp = other->backOp;
    // Copy requires_grad flag
    tensor->requires_grad = other->requires_grad;

    return tensor;
}


/*
    Entry Function for backward propagation.
    Accumulates incoming gradient and propagates it backward through the creating operation (`backOp`),
    but only if this tensor requires gradients.
*/
void tensor_float_backward(Tensor_float* self, Matrix_float* grad_incoming) 
{
    assert(self && grad_incoming && "Tensor and incoming grad cannot be NULL for backward");

    // Only proceed if gradients are required for this tensor
    if (!self->requires_grad) 
    {
        return; // Do nothing if grads are not needed
    }

    // Ensure gradient matrix exists (should have been created by zeroGrad if requires_grad is true)
    assert(self->grad && "Tensor requires grad, but grad matrix is NULL");
    assert(self->val && "Tensor value matrix is NULL in backward");

    // Verify shapes match for accumulation
    assert(matrix_float_verifyShapeForElementwiseOperation(self->val->shape, self->val->num_dims, grad_incoming->shape, grad_incoming->num_dims) &&
           "The incoming gradient and the tensor value shapes do not match !");
    assert(matrix_float_verifyShapeForElementwiseOperation(self->grad->shape, self->grad->num_dims, grad_incoming->shape, grad_incoming->num_dims) &&
           "The incoming gradient and the tensor gradient shapes do not match !");

    // Accumulate gradient: this->grad = this->grad + grad_incoming;
    Matrix_float* temp_sum = matrix_float_add(self->grad, grad_incoming);
    matrix_float_destroy(self->grad); // Free the old gradient matrix
    self->grad = temp_sum;            // Assign the new summed gradient matrix

    // Propagate backward if there's a backward operation and it has a backward function
    if (self->backOp != NULL && self->backOp->backward != NULL) 
    {
        // Pass the *accumulated* gradient backward
        self->backOp->backward(self->backOp, self->grad); // Pass self->grad
    }
    // Note: grad_incoming is NOT freed here, its ownership depends on the caller.
}

/*
    Starts the backward pass from this tensor, assuming the gradient of the final output
    with respect to this tensor is 1. Creates a matrix of ones as the initial incoming gradient.
*/
void tensor_float_backward_default(Tensor_float* self) 
{
    assert(self && self->val && "Tensor and its value matrix must exist for default backward");

    // Only proceed if gradients are required
    if (!self->requires_grad) 
    {
        fprintf(stderr, "Warning: backward_default called on tensor that does not require grad.\n");
        return;
    }

    // Make gradient of all 1's matching the shape of val
    float* ones_data = (float*)malloc(self->val->num_elements * sizeof(float));
    assert(ones_data != NULL);
    for (size_t i = 0; i < self->val->num_elements; ++i) 
    {
        ones_data[i] = 1.0f;
    }
    Matrix_float* grad_ones = matrix_float_create(ones_data, self->val->shape, self->val->num_dims);
    free(ones_data); // create copies data

    // Call the main backward function with this gradient of ones
    tensor_float_backward(self, grad_ones);

    // Destroy the temporary gradient of ones as it's no longer needed
    matrix_float_destroy(grad_ones);
}


/*
    Destructor for Tensor. Frees the owned matrices (`val`, `grad`).
    Does NOT free the Operation pointers (`frontOp`, `backOp`) as per the revised ownership model.
*/
void tensor_float_destroy(Tensor_float* tensor) 
{
    if (tensor) 
    {
        // Free owned data
        matrix_float_destroy(tensor->val);
        matrix_float_destroy(tensor->grad); // Safe even if grad is NULL

        // Do NOT destroy backOp or frontOp here.
        // They are managed elsewhere (e.g., by a graph structure or manually).

        // Free the tensor struct itself
        free(tensor);
    }
}
// ========================================
// END OF "types/tensor.h" content
// ========================================


// ========================================
// START OF SPECIFIC OPERATION HEADERS (.h parts)
// ========================================

// --- AddOperation ---
struct AddOperation_float 
{
    Operation_float base_op; // MUST be first member for casting
    // AddOperation specific members (none needed here)
};
AddOperation_float* AddOperation_float_create(Tensor_float* t1, Tensor_float* t2);
// Implementations (_backward, _forward, _destroy) defined later


// --- MultiplyOperation ---
struct MultiplyOperation_float 
{
    Operation_float base_op; // MUST be first member
};
MultiplyOperation_float* MultiplyOperation_float_create(Tensor_float* t1, Tensor_float* t2);
// Implementations (_backward, _forward, _destroy) defined later


// --- DivideOperation ---
struct DivideOperation_float 
{
    Operation_float base_op; // MUST be first member
};
DivideOperation_float* DivideOperation_float_create(Tensor_float* t1, Tensor_float* t2);
// Implementations (_backward, _forward, _destroy) defined later


// --- ExponentOperation ---
struct ExponentOperation_float 
{
    Operation_float base_op; // MUST be first member
    // No t2 for unary ops
};
ExponentOperation_float* ExponentOperation_float_create(Tensor_float* t1);
// Implementations (_backward, _forward, _destroy) defined later


// --- DotOperation ---
struct DotOperation_float 
{
    Operation_float base_op; // MUST be first member
};
DotOperation_float* DotOperation_float_create(Tensor_float* t1, Tensor_float* t2);
// Implementations (_backward, _forward, _destroy) defined later


// --- SigmoidOperation ---
struct SigmoidOperation_float 
{
    Operation_float base_op; // MUST be first member
};
SigmoidOperation_float* SigmoidOperation_float_create(Tensor_float* t1);
// Implementations (_backward, _forward, _destroy) defined later


// --- PowerOperation ---
struct PowerOperation_float 
{
    Operation_float base_op; // MUST be first member
    float pow_val; // Stores the exponent scalar
};
PowerOperation_float* PowerOperation_float_create(Tensor_float* t1, float pow_val);
// Implementations (_backward, _forward, _destroy) defined later

// ========================================
// END OF SPECIFIC OPERATION HEADERS (.h parts)
// ========================================


// ========================================
// START OF "operations/operations_impl.h" content (Specific Operation Implementations)
// ========================================

// --- AddOperation Implementation ---
void AddOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
    assert(self && self->t1 && self->t2 && grad && "NULL pointer in AddOperation backward");
    // Addition distributes gradient equally. Propagate only if operand requires grad.
    if (self->t1->requires_grad) 
    {
         tensor_float_backward(self->t1, grad); // Pass same grad matrix
    }
    if (self->t2->requires_grad) 
    {
         tensor_float_backward(self->t2, grad); // Pass same grad matrix
    }
    // Note: grad is not freed here, it belongs to the result tensor (t3).
}

Tensor_float* AddOperation_float_forward(Operation_float* self) 
{
    assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && "NULL pointer in AddOperation forward");
    Matrix_float* result_val = matrix_float_add(self->t1->val, self->t2->val);
    // Result requires grad if either input requires grad
    bool requires_grad = self->t1->requires_grad || self->t2->requires_grad;
    // The operation creates the result tensor and sets itself as backOp
    self->t3 = tensor_float_create_from_op(result_val, self, requires_grad); // create takes ownership of result_val
    // Link the operation back to the result tensor
    self->t3->backOp = self;
    return self->t3;
}

void AddOperation_float_destroy(Operation_float* self) 
{
    if (self) 
    {
        // Operation owns its result tensor t3. Destroy it.
        tensor_float_destroy(self->t3); // Assumes tensor_destroy frees matrices but not ops
        // operation_float_base_destroy(self); // Base currently does nothing useful
        free(self); // Free the operation struct itself
    }
}

AddOperation_float* AddOperation_float_create(Tensor_float* t1, Tensor_float* t2) 
{
    AddOperation_float* op = (AddOperation_float*)malloc(sizeof(AddOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = t2;
    op->base_op.t3 = NULL; // Result tensor created during forward pass
    op->base_op.backward = AddOperation_float_backward;
    op->base_op.forward = AddOperation_float_forward;
    op->base_op.destroy = AddOperation_float_destroy;
    return op;
}


// --- MultiplyOperation Implementation ---
void MultiplyOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
    assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && grad && "NULL pointer in MultiplyOperation backward");

    if (self->t1->requires_grad) 
    {
        // Gradient for t1: grad * t2->val
        Matrix_float* grad_t1 = matrix_float_multiply_elementwise(grad, self->t2->val);
        tensor_float_backward(self->t1, grad_t1);
        matrix_float_destroy(grad_t1); // grad_t1 is temporary for this path
    }

    if (self->t2->requires_grad) 
    {
        // Gradient for t2: grad * t1->val
        Matrix_float* grad_t2 = matrix_float_multiply_elementwise(grad, self->t1->val);
        tensor_float_backward(self->t2, grad_t2);
        matrix_float_destroy(grad_t2); // grad_t2 is temporary for this path
    }
}

Tensor_float* MultiplyOperation_float_forward(Operation_float* self) 
{
    assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && "NULL pointer in MultiplyOperation forward");
    Matrix_float* result_val = matrix_float_multiply_elementwise(self->t1->val, self->t2->val);
    bool requires_grad = self->t1->requires_grad || self->t2->requires_grad;
    self->t3 = tensor_float_create_from_op(result_val, self, requires_grad);
    self->t3->backOp = self;
    return self->t3;
}

void MultiplyOperation_float_destroy(Operation_float* self) 
{
     if (self) 
     {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
     }
}

MultiplyOperation_float* MultiplyOperation_float_create(Tensor_float* t1, Tensor_float* t2) 
{
    MultiplyOperation_float* op = (MultiplyOperation_float*)malloc(sizeof(MultiplyOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = t2;
    op->base_op.t3 = NULL;
    op->base_op.backward = MultiplyOperation_float_backward;
    op->base_op.forward = MultiplyOperation_float_forward;
    op->base_op.destroy = MultiplyOperation_float_destroy;
    return op;
}


// --- DivideOperation Implementation ---
// d(t1/t2)/dt1 = 1/t2
// d(t1/t2)/dt2 = -t1 / (t2*t2)
void DivideOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
     assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && grad && "NULL pointer in DivideOperation backward");

    if (self->t1->requires_grad) 
    {
        // grad_t1 = grad * (1 / t2->val)
        Matrix_float* one_over_t2 = matrix_float_divide_scalar_lhs(1.0f, self->t2->val); // Checks for 0 in t2
        Matrix_float* grad_t1 = matrix_float_multiply_elementwise(grad, one_over_t2);
        tensor_float_backward(self->t1, grad_t1);
        // Cleanup intermediates for this path
        matrix_float_destroy(one_over_t2);
        matrix_float_destroy(grad_t1);
    }

    if (self->t2->requires_grad) 
    {
        // grad_t2 = grad * (-t1->val / (t2->val * t2->val))
        Matrix_float* t2_squared = matrix_float_multiply_elementwise(self->t2->val, self->t2->val);
        Matrix_float* minus_t1 = matrix_float_multiply_scalar(-1.0f, self->t1->val);
        Matrix_float* factor_t2 = matrix_float_divide_elementwise(minus_t1, t2_squared); // Checks for 0 in t2_squared
        Matrix_float* grad_t2 = matrix_float_multiply_elementwise(grad, factor_t2);
        tensor_float_backward(self->t2, grad_t2);
        // Cleanup intermediates for this path
        matrix_float_destroy(t2_squared);
        matrix_float_destroy(minus_t1);
        matrix_float_destroy(factor_t2);
        matrix_float_destroy(grad_t2);
    }
}

Tensor_float* DivideOperation_float_forward(Operation_float* self) 
{
    assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && "NULL pointer in DivideOperation forward");
    // matrix_divide_elementwise checks for division by zero inside vector_divide_vv
    Matrix_float* result_val = matrix_float_divide_elementwise(self->t1->val, self->t2->val);
    bool requires_grad = self->t1->requires_grad || self->t2->requires_grad;
    self->t3 = tensor_float_create_from_op(result_val, self, requires_grad);
    self->t3->backOp = self;
    return self->t3;
}

void DivideOperation_float_destroy(Operation_float* self) 
{
     if (self) 
     {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
     }
}

DivideOperation_float* DivideOperation_float_create(Tensor_float* t1, Tensor_float* t2) 
{
    DivideOperation_float* op = (DivideOperation_float*)malloc(sizeof(DivideOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = t2;
    op->base_op.t3 = NULL;
    op->base_op.backward = DivideOperation_float_backward;
    op->base_op.forward = DivideOperation_float_forward;
    op->base_op.destroy = DivideOperation_float_destroy;
    return op;
}


// --- ExponentOperation Implementation ---
// d(exp(t1))/dt1 = exp(t1) = result (t3)
void ExponentOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
    // self->t3 must exist if backward is called after forward
    assert(self && self->t1 && self->t1->val && self->t3 && self->t3->val && grad && "NULL pointer in ExponentOperation backward");
    // Only propagate if t1 requires grad
    if (self->t1->requires_grad) 
    {
        // Gradient is grad * exp(t1), which is grad * result_value (t3->val)
        Matrix_float* grad_t1 = matrix_float_multiply_elementwise(grad, self->t3->val);
        tensor_float_backward(self->t1, grad_t1);
        matrix_float_destroy(grad_t1); // Temporary gradient
     }
}

Tensor_float* ExponentOperation_float_forward(Operation_float* self) 
{
    assert(self && self->t1 && self->t1->val && "NULL pointer in ExponentOperation forward");
    Matrix_float* result_val = matrix_float_exp(self->t1->val);
    // Result requires grad if input requires grad
    bool requires_grad = self->t1->requires_grad;
    self->t3 = tensor_float_create_from_op(result_val, self, requires_grad);
    self->t3->backOp = self;
    return self->t3;
}

void ExponentOperation_float_destroy(Operation_float* self) 
{
    if (self) 
    {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
    }
}

ExponentOperation_float* ExponentOperation_float_create(Tensor_float* t1) 
{
    ExponentOperation_float* op = (ExponentOperation_float*)malloc(sizeof(ExponentOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = NULL; // Unary operation
    op->base_op.t3 = NULL;
    op->base_op.backward = ExponentOperation_float_backward;
    op->base_op.forward = ExponentOperation_float_forward;
    op->base_op.destroy = ExponentOperation_float_destroy;
    return op;
}


// --- DotOperation Implementation ---
// Need matrix transpose utility function first
// --- Utility Function ---
/*
    Function to convert a Matrix to a transpose of it.
    Currently only supports 2D matrix transpose.
    Creates a new matrix, caller must free.
*/
Matrix_float* utils_matrix_transpose_2d(const Matrix_float* m) 
{
    assert(m && m->num_dims == 2 && "Transpose currently only supports 2D matrices");

    int rows = m->shape[0];
    int cols = m->shape[1];
    int new_shape[] = {cols, rows};
    float* transposed_val = (float*)malloc(m->num_elements * sizeof(float));
    assert(transposed_val != NULL);

    for (int i = 0; i < rows; ++i) 
    {
        for (int j = 0; j < cols; ++j) 
        {
            // transposed[j, i] = original[i, j]
            transposed_val[(size_t)j * rows + i] = m->val[(size_t)i * cols + j];
        }
    }

    Matrix_float* transposed_matrix = matrix_float_create(transposed_val, new_shape, 2);
    free(transposed_val); // create copies data
    return transposed_matrix;
}

// d(t1.dot(t2)) / dt1 = grad.dot(t2^T)
// d(t1.dot(t2)) / dt2 = (t1^T).dot(grad)
// Assuming t1 = (B, N), t2 = (N, M), grad = (B, M)
// grad_t1 = (B, M) dot (M, N) = (B, N)
// grad_t2 = (N, B) dot (B, M) = (N, M)
// NOTE: This implementation assumes 2D matrices for simplicity, matching common NN layers.
//       Extending to ND . ND backward is significantly more complex.
void DotOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
     assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && grad && "NULL pointer in DotOperation backward");
     // Assuming ND matmul where only last 2 dims of t1 and the 2 dims of t2 participate.
     // grad will have shape corresponding to result: (t1.shape[:-2], t1.shape[-2], t2.shape[-1])
     // This simple 2D backward implementation won't work correctly for ND inputs without adjustments.
     assert(self->t1->val->num_dims == 2 && self->t2->val->num_dims == 2 && grad->num_dims == 2 && "Dot backward currently assumes 2D matrices for t1, t2, and grad");

    if (self->t1->requires_grad) 
    {
        // Calculate grad_t1 = grad .dot( t2^T )
        Matrix_float* t2_T = utils_matrix_transpose_2d(self->t2->val);
        Matrix_float* grad_t1 = matrix_float_dot(grad, t2_T);
        tensor_float_backward(self->t1, grad_t1);
        // Cleanup intermediates for this path
        matrix_float_destroy(t2_T);
        matrix_float_destroy(grad_t1);
    }

    if (self->t2->requires_grad) 
    {
        // Calculate grad_t2 = ( t1^T ) .dot( grad )
        Matrix_float* t1_T = utils_matrix_transpose_2d(self->t1->val);
        Matrix_float* grad_t2 = matrix_float_dot(t1_T, grad);
        tensor_float_backward(self->t2, grad_t2);
        // Cleanup intermediates for this path
        matrix_float_destroy(t1_T);
        matrix_float_destroy(grad_t2);
    }
}

Tensor_float* DotOperation_float_forward(Operation_float* self) 
{
    assert(self && self->t1 && self->t2 && self->t1->val && self->t2->val && "NULL pointer in DotOperation forward");
    // matrix_float_dot checks compatibility
    Matrix_float* result_val = matrix_float_dot(self->t1->val, self->t2->val);
    bool requires_grad = self->t1->requires_grad || self->t2->requires_grad;
    self->t3 = tensor_float_create_from_op(result_val, self, requires_grad);
    self->t3->backOp = self;
    return self->t3;
}

void DotOperation_float_destroy(Operation_float* self) 
{
     if (self) 
     {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
     }
}

DotOperation_float* DotOperation_float_create(Tensor_float* t1, Tensor_float* t2) 
{
    DotOperation_float* op = (DotOperation_float*)malloc(sizeof(DotOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = t2;
    op->base_op.t3 = NULL;
    op->base_op.backward = DotOperation_float_backward;
    op->base_op.forward = DotOperation_float_forward;
    op->base_op.destroy = DotOperation_float_destroy;
    return op;
}


// --- SigmoidOperation Implementation ---
// s = sigmoid(t1) = 1 / (1 + exp(-t1)) = t3
// d(s)/dt1 = s * (1 - s) = t3 * (1 - t3)
void SigmoidOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
    // self->t3 must exist if backward is called after forward
    assert(self && self->t1 && self->t1->val && self->t3 && self->t3->val && grad && "NULL pointer in SigmoidOperation backward");

    if (self->t1->requires_grad) 
    {
        // Calculate derivative: t3->val * (1 - t3->val)
        Matrix_float* one_minus_t3 = matrix_float_subtract_scalar_lhs(1.0f, self->t3->val);
        Matrix_float* derivative = matrix_float_multiply_elementwise(self->t3->val, one_minus_t3);

        // Calculate gradient for t1: grad * derivative
        Matrix_float* grad_t1 = matrix_float_multiply_elementwise(grad, derivative);
        tensor_float_backward(self->t1, grad_t1);

        // Cleanup
        matrix_float_destroy(one_minus_t3);
        matrix_float_destroy(derivative);
        matrix_float_destroy(grad_t1);
    }
}

// Sigmoid forward uses matrixOps_sigmoid_float helper
Matrix_float* matrixOps_sigmoid_float(const Matrix_float* a) 
{
    assert(a != NULL);
    // Calculate: 1 / (1 + exp(-1 * a))
    Matrix_float* minus_a = matrix_float_multiply_scalar(-1.0f, a);
    Matrix_float* exp_minus_a = matrix_float_exp(minus_a);
    Matrix_float* one_plus_exp = matrix_float_add_scalar(1.0f, exp_minus_a);
    Matrix_float* result = matrix_float_divide_scalar_lhs(1.0f, one_plus_exp); // Checks for 0 denominator inside

    // Cleanup intermediates
    matrix_float_destroy(minus_a);
    matrix_float_destroy(exp_minus_a);
    matrix_float_destroy(one_plus_exp);

    return result;
}


Tensor_float* SigmoidOperation_float_forward(Operation_float* self) 
{
     assert(self && self->t1 && self->t1->val && "NULL pointer in SigmoidOperation forward");
     Matrix_float* result_val = matrixOps_sigmoid_float(self->t1->val);
     bool requires_grad = self->t1->requires_grad;
     self->t3 = tensor_float_create_from_op(result_val, self, requires_grad);
     self->t3->backOp = self;
     return self->t3;
}

void SigmoidOperation_float_destroy(Operation_float* self) 
{
     if (self) 
     {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
     }
}

SigmoidOperation_float* SigmoidOperation_float_create(Tensor_float* t1) 
{
    SigmoidOperation_float* op = (SigmoidOperation_float*)malloc(sizeof(SigmoidOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = NULL;
    op->base_op.t3 = NULL;
    op->base_op.backward = SigmoidOperation_float_backward;
    op->base_op.forward = SigmoidOperation_float_forward;
    op->base_op.destroy = SigmoidOperation_float_destroy;
    return op;
}


// --- PowerOperation Implementation ---
// p = t1 ^ pow_val = t3
// d(p)/dt1 = pow_val * (t1 ^ (pow_val - 1))
void PowerOperation_float_backward(Operation_float* self, Matrix_float* grad) 
{
    // Cast self to access pow_val
    PowerOperation_float* op = (PowerOperation_float*)self;
    assert(op && op->base_op.t1 && op->base_op.t1->val && grad && "NULL pointer in PowerOperation backward");

    if (op->base_op.t1->requires_grad) 
    {
        // Calculate derivative: pow_val * (t1->val ^ (pow_val - 1))
        // Handle potential issues with pow(negative_base, non_integer_exponent-1) if necessary
        Matrix_float* t1_pow_minus_1 = matrix_float_power_scalar(op->base_op.t1->val, op->pow_val - 1.0f);
        Matrix_float* derivative = matrix_float_multiply_scalar(op->pow_val, t1_pow_minus_1);

        // Calculate gradient for t1: grad * derivative
        Matrix_float* grad_t1 = matrix_float_multiply_elementwise(grad, derivative);
        tensor_float_backward(op->base_op.t1, grad_t1);

        // Cleanup
        matrix_float_destroy(t1_pow_minus_1);
        matrix_float_destroy(derivative);
        matrix_float_destroy(grad_t1);
    }
}

// Power forward uses matrix_float_power_scalar helper defined earlier
Matrix_float* matrixOps_power_float(Matrix_float* a, float pow_val) 
{
    // Ensure input is not const if matrix_float_power_scalar expects non-const
    // Current signature is (const Matrix_float*, float), so it's okay.
    return matrix_float_power_scalar(a, pow_val);
}


Tensor_float* PowerOperation_float_forward(Operation_float* self) 
{
    PowerOperation_float* op = (PowerOperation_float*)self;
    assert(op && op->base_op.t1 && op->base_op.t1->val && "NULL pointer in PowerOperation forward");
    // Cast t1->val to const for power_scalar function if needed, or change power_scalar signature
    Matrix_float* result_val = matrix_float_power_scalar(op->base_op.t1->val, op->pow_val);
    bool requires_grad = op->base_op.t1->requires_grad;
    op->base_op.t3 = tensor_float_create_from_op(result_val, (Operation_float*)op, requires_grad);
    op->base_op.t3->backOp = (Operation_float*)op;
    return op->base_op.t3;
}

void PowerOperation_float_destroy(Operation_float* self) 
{
     if (self) 
     {
        tensor_float_destroy(self->t3);
        // operation_float_base_destroy(self);
        free(self);
     }
}

PowerOperation_float* PowerOperation_float_create(Tensor_float* t1, float pow_val) 
{
    PowerOperation_float* op = (PowerOperation_float*)malloc(sizeof(PowerOperation_float));
    assert(op != NULL);
    op->base_op.t1 = t1;
    op->base_op.t2 = NULL;
    op->base_op.t3 = NULL;
    op->pow_val = pow_val; // Store the exponent
    op->base_op.backward = PowerOperation_float_backward;
    op->base_op.forward = PowerOperation_float_forward;
    op->base_op.destroy = PowerOperation_float_destroy;
    return op;
}
// ========================================
// END OF "operations/operations_impl.h" content
// ========================================


// ========================================
// START OF "overloads/tensor.h" content (tensorOps namespace)
// ========================================
/*
    This file defines the various mathematical operations of the Tensor Struct.
    (Emulated via tensorOps_ prefix)

    These functions create the appropriate Operation struct, link the input tensors
    to it (`frontOp`), execute the forward pass (which creates the result tensor
    and links it via `backOp`), and return the result tensor.

    Memory Management Note: The operation struct created here (e.g., add_op_specific)
    is now owned by the computational graph implicitly. The result tensor it creates (`t3`)
    is owned by the operation. The input tensors (`one`, `two`) have their `frontOp`
    pointer set, but they don't own the operation. A separate mechanism is needed
    to eventually destroy the operations (e.g., when the graph is no longer needed).
*/

// --- Helper for Temporary Scalar Tensors ---
// Creates a tensor with the same shape as 'like_tensor', filled with 'scalar'.
// The caller must eventually destroy this temporary tensor.
// This tensor does NOT require gradients.
Tensor_float* tensorOps_create_scalar_tensor_like(float scalar, const Tensor_float* like_tensor) 
{
    assert(like_tensor && like_tensor->val);
    size_t num_elements = like_tensor->val->num_elements;
    float* scalar_data = (float*)malloc(num_elements * sizeof(float));
    assert(scalar_data != NULL);
    for(size_t i=0; i < num_elements; ++i) scalar_data[i] = scalar;
    // Scalar tensors typically don't require gradients themselves
    Tensor_float* scalar_tensor = tensor_float_create_from_values(scalar_data, like_tensor->val->shape, like_tensor->val->num_dims, false);
    free(scalar_data); // create copies data
    return scalar_tensor;
}


// Addition
Tensor_float* tensorOps_add_float(Tensor_float* one, Tensor_float* two) 
{
    assert(one && two);
    // Create the AddOperation
    AddOperation_float* add_op_specific = AddOperation_float_create(one, two);
    Operation_float* op = (Operation_float*)add_op_specific; // Cast to base pointer
    // Link tensors to the operation (they are operands)
    one->frontOp = op;
    two->frontOp = op;
    // Execute forward pass, which creates the result tensor and links it via backOp
    Tensor_float* result = op->forward(op);
    // The operation 'op' is now part of the graph, linked via result->backOp.
    // It should be destroyed later by graph cleanup, not here.
    return result;
}

// Addition with Scalar (scalar + tensor)
Tensor_float* tensorOps_add_scalar_float(float scalar, Tensor_float* two) 
{
    assert(two && two->val);
    // Create a temporary tensor for the scalar
    Tensor_float* one = tensorOps_create_scalar_tensor_like(scalar, two);

    // Perform tensor-tensor addition
    Tensor_float* result = tensorOps_add_float(one, two);

    // IMPORTANT: The temporary scalar tensor 'one' is now linked into the graph
    // via result->backOp->t1. It MUST NOT be destroyed here.
    // It will be destroyed when the graph/operation is destroyed, assuming
    // the Op->destroy handles its inputs correctly (which it currently doesn't).
    // The current Op->destroy only handles t3.
    // This highlights a remaining memory management complexity.
    // A possible fix: Ops don't destroy input tensors t1/t2. A graph manager does.

    // Alternative: If 'one' is truly temporary and known not to be needed elsewhere,
    // and if AddOperation didn't store a pointer to it permanently (which it does),
    // then it *could* be freed. But the current graph structure makes this unsafe.
    // tensor_float_destroy(one); // <-- Very Unsafe with current structure

    return result;
}
// Addition with Scalar (tensor + scalar) - Convenience wrapper
Tensor_float* tensorOps_add_scalar_rhs_float(Tensor_float* one, float scalar) 
{
    return tensorOps_add_scalar_float(scalar, one); // Reuse the other function
}


// Divide (tensor / tensor)
Tensor_float* tensorOps_divide_float(Tensor_float* one, Tensor_float* two) 
{
    assert(one && two);
    DivideOperation_float* div_op_specific = DivideOperation_float_create(one, two);
    Operation_float* op = (Operation_float*)div_op_specific;
    one->frontOp = op;
    two->frontOp = op;
    return op->forward(op);
}

// Divide Scalar (scalar / tensor)
Tensor_float* tensorOps_divide_scalar_lhs_float(float scalar, Tensor_float* two) 
{
    assert(two && two->val);
    Tensor_float* one = tensorOps_create_scalar_tensor_like(scalar, two);
    Tensor_float* result = tensorOps_divide_float(one, two);
    // tensor_float_destroy(one); // Unsafe - part of graph now
    return result;
}

// Divide Scalar (tensor / scalar)
Tensor_float* tensorOps_divide_scalar_rhs_float(Tensor_float* one, float scalar) 
{
    assert(one && one->val);
    assert(scalar != 0.0f && "Division by zero scalar");
    Tensor_float* two = tensorOps_create_scalar_tensor_like(scalar, one);
    Tensor_float* result = tensorOps_divide_float(one, two);
    // tensor_float_destroy(two); // Unsafe - part of graph now
    return result;
}


// Multiply (tensor * tensor)
Tensor_float* tensorOps_multiply_float(Tensor_float* one, Tensor_float* two) 
{
    assert(one && two);
    MultiplyOperation_float* mul_op_specific = MultiplyOperation_float_create(one, two);
    Operation_float* op = (Operation_float*)mul_op_specific;
    one->frontOp = op;
    two->frontOp = op;
    return op->forward(op);
}

// Multiply with scalar (scalar * tensor)
Tensor_float* tensorOps_multiply_scalar_float(float scalar, Tensor_float* two) 
{
    assert(two && two->val);
    Tensor_float* one = tensorOps_create_scalar_tensor_like(scalar, two);
    Tensor_float* result = tensorOps_multiply_float(one, two);
    // tensor_float_destroy(one); // Unsafe
    return result;
}
// Multiply with scalar (tensor * scalar) - Convenience wrapper
Tensor_float* tensorOps_multiply_scalar_rhs_float(Tensor_float* one, float scalar) 
{
    return tensorOps_multiply_scalar_float(scalar, one);
}


// Dot Product (tensor . tensor)
Tensor_float* tensorOps_dot_float(Tensor_float* one, Tensor_float* two) 
{
    assert(one && two);
    DotOperation_float* dot_op_specific = DotOperation_float_create(one, two);
    Operation_float* op = (Operation_float*)dot_op_specific;
    one->frontOp = op;
    two->frontOp = op;
    return op->forward(op);
}

// Exponent (exp(tensor))
Tensor_float* tensorOps_exp_float(Tensor_float* one) 
{
    assert(one);
    ExponentOperation_float* exp_op_specific = ExponentOperation_float_create(one);
    Operation_float* op = (Operation_float*)exp_op_specific;
    one->frontOp = op;
    return op->forward(op);
}

// Sigmoid (sigmoid(tensor))
Tensor_float* tensorOps_sigmoid_float(Tensor_float* one) 
{
    assert(one);
    SigmoidOperation_float* sig_op_specific = SigmoidOperation_float_create(one);
    Operation_float* op = (Operation_float*)sig_op_specific;
    one->frontOp = op;
    return op->forward(op);
}

// Power (tensor ^ scalar)
Tensor_float* tensorOps_power_float(Tensor_float* one, float scalar_pow) 
{
    assert(one);
    PowerOperation_float* pow_op_specific = PowerOperation_float_create(one, scalar_pow);
    Operation_float* op = (Operation_float*)pow_op_specific;
    one->frontOp = op;
    return op->forward(op);
}

// ========================================
// END OF "overloads/tensor.h" content
// ========================================


// ========================================
// START OF "utils/weights.h" content
// ========================================
/*
    Utils for weights. Different weight initialisation schemes and for debugging weights in the future
*/

/*
    Glorot/ Xavier Initialisation (Uniform). See paper:
    "Understanding the difficulty of training deep feedforward neural networks"
    by Bengio and Glorot
    Paper Link: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Using a uniform distribution U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    which has variance = 2 / (fan_in + fan_out).
    Returns a dynamically allocated array, caller must free.
*/
float* utils_glorotInit_float(int fan_in, int fan_out) 
{
    assert(fan_in > 0 && fan_out > 0);
    // double variance = 2.0 / (double)(fan_in + fan_out); // Variance calculation not directly needed for uniform limit
    // double stddev = sqrt(variance); // Unused variable removed
    // For uniform distribution U(-a, a), variance is a^2 / 3.
    // We want variance = 2/(fin+fout) => a^2/3 = 2/(fin+fout) => a = sqrt(6/(fin+fout))
    double limit = sqrt(6.0 / (double)(fan_in + fan_out));

    size_t num_weights = (size_t)fan_in * fan_out;
    float* weights = (float*)malloc(num_weights * sizeof(float));
    assert(weights != NULL);

    // Seed random number generator (should ideally be done once at program start, e.g., in main)
    // srand(time(NULL)); // Example seeding - Avoid calling srand frequently

    for (size_t i = 0; i < num_weights; i++) 
    {
        // Generate random double between 0.0 and 1.0
        double random_val_01 = (double)rand() / (double)RAND_MAX;
        // Scale to [-limit, limit]
        // random_val_01 * 2.0 -> [0.0, 2.0]
        // (random_val_01 * 2.0 - 1.0) -> [-1.0, 1.0]
        // (...) * limit -> [-limit, limit]
        weights[i] = (float)((random_val_01 * 2.0 - 1.0) * limit);
    }

    return weights;
}
// ========================================
// END OF "utils/weights.h" content
// ========================================


// ========================================
// START OF "utils/matrix.h" content
// ========================================
/*
    Utils for matrices. Functions like transpose and for future util matrix functions.
*/

/*
    Function to convert a 2D Matrix to its transpose.
    Creates a new matrix (deep copy), caller must free.
*/
// (utils_matrix_transpose_2d already defined within DotOperation implementation section, reuse that)


/*
    Creates a new matrix of the same shape as m, filled with zeros.
    Caller must free the returned matrix.
*/
Matrix_float* utils_matrix_zerosLike(const Matrix_float* m) 
{
    assert(m && m->shape && m->num_dims > 0);
    float* zero_data = (float*)calloc(m->num_elements, sizeof(float));
    assert(zero_data != NULL);
    Matrix_float* zero_matrix = matrix_float_create(zero_data, m->shape, m->num_dims);
    free(zero_data); // create copies data
    return zero_matrix;
}
// ========================================
// END OF "utils/matrix.h" content
// ========================================


// ========================================
// START OF "optims/optim.h" content
// ========================================
/*
    This file defines the Base Struct for all Optimizers.
*/

// --- Dynamic Array for Tensor Pointers (Set Emulation) ---
// Used to store unique tensor pointers (parameters) that the optimizer manages.
typedef struct TensorPointerSet 
{
    Tensor_float** items; // Array of tensor pointers
    size_t count;         // Number of items currently in the set
    size_t capacity;      // Allocated size of the items array
} TensorPointerSet;

// Creates a new, empty TensorPointerSet.
TensorPointerSet* tensor_pointer_set_create(size_t initial_capacity) 
{
    TensorPointerSet* set = (TensorPointerSet*)malloc(sizeof(TensorPointerSet));
    assert(set);
    set->capacity = initial_capacity > 0 ? initial_capacity : 16; // Default capacity 16
    set->items = (Tensor_float**)malloc(set->capacity * sizeof(Tensor_float*));
    assert(set->items);
    set->count = 0;
    return set;
}

// Destroys a TensorPointerSet, freeing the internal array but NOT the tensors it points to.
void tensor_pointer_set_destroy(TensorPointerSet* set) 
{
    if (set) 
    {
        free(set->items);
        free(set);
    }
}

// Checks if a tensor pointer already exists in the set (linear search).
bool tensor_pointer_set_contains(TensorPointerSet* set, Tensor_float* tensor) 
{
    for (size_t i = 0; i < set->count; ++i) 
    {
        if (set->items[i] == tensor) 
        {
            return true;
        }
    }
    return false;
}

// Adds a tensor pointer to the set if it's not already present. Resizes if necessary.
void tensor_pointer_set_add(TensorPointerSet* set, Tensor_float* tensor) 
{
    if (!tensor) return; // Don't add NULL pointers
    if (tensor_pointer_set_contains(set, tensor)) 
    {
        return; // Already exists
    }
    // Resize if full
    if (set->count >= set->capacity) 
    {
        set->capacity *= 2;
        set->items = (Tensor_float**)realloc(set->items, set->capacity * sizeof(Tensor_float*));
        assert(set->items);
    }
    // Add the new item
    set->items[set->count++] = tensor;
}

// Removes all items from the set (sets count to 0). Does not shrink capacity.
void tensor_pointer_set_clear(TensorPointerSet* set) 
{
    if (set) 
    {
        set->count = 0;
    }
}
// --- End Set Emulation ---


// Base Optimizer struct
struct Optimizer_float 
{
    // This set contains pointers to all the Tensors (parameters) that this optimizer should update.
    TensorPointerSet* params;

    // The learning rate.
    float lr;

    // --- Function pointers for polymorphism (emulating virtual methods) ---
    // Performs one optimization step (e.g., gradient descent update).
    void (*step)(Optimizer_float* self); // Changed signature: LR is member
    // Finds and collects parameters from a computational graph starting from tensor x.
    void (*getParams)(Optimizer_float* self, Tensor_float* x);
    // Zeros the gradients of all managed parameters.
    void (*zeroGrad)(Optimizer_float* self);
    // Destroys the optimizer and its internal structures (like the params set).
    void (*destroy)(Optimizer_float* self);

    // Add other optimizer-specific members here or in derived structs (e.g., momentum terms)
};

// Base initializer (called by specific optimizer create functions)
void optimizer_float_init_base(Optimizer_float* self, float lr) 
{
    self->params = tensor_pointer_set_create(16); // Initial capacity for params
    self->lr = lr;
    // Function pointers must be set by the derived optimizer's create function
    self->step = NULL;
    self->getParams = NULL;
    self->zeroGrad = NULL; // Default zeroGrad can be provided
    self->destroy = NULL;
}

// Base destructor helper (called by specific optimizer destroy functions)
void optimizer_float_destroy_base(Optimizer_float* self) 
{
    if (self) 
    {
        // Destroy the parameter set (frees the array, not the tensors)
        tensor_pointer_set_destroy(self->params);
        // Don't free 'self' here, the derived destroyer does that.
    }
}

// Default implementation for zeroing gradients. Can be assigned to self->zeroGrad.
void optimizer_float_default_zeroGrad(Optimizer_float* self) 
{
    assert(self && self->params);
    for (size_t i = 0; i < self->params->count; i++) 
    {
        Tensor_float* param = self->params->items[i];
        if (param && param->requires_grad) 
        {
            tensor_float_zeroGrad(param); // Call tensor's zeroGrad function
        }
    }
}
// ========================================
// END OF "optims/optim.h" content
// ========================================


// ========================================
// START OF "optims/sgd.h" content
// ========================================
/*
    This file defines the Stochastic Gradient Descent Optimiser.

    It finds parameters in the graph (`getParams`), updates them using their gradients
    (`step`), and can zero gradients (`zeroGrad`).

    Update Rule (in `step`): param_value = param_value - learning_rate * param_gradient
*/

// --- Queue Emulation for BFS (used in getParams) ---
// Simple dynamic array based queue for Tensor pointers.
typedef struct TensorPointerQueue 
{
    Tensor_float** items; // Array of tensor pointers
    size_t head;          // Index of the front item
    size_t tail;          // Index *after* the last item
    size_t count;         // Number of items in the queue
    size_t capacity;      // Allocated size of the items array
} TensorPointerQueue;

// Creates a new, empty queue.
TensorPointerQueue* tensor_pointer_queue_create(size_t initial_capacity) 
{
    TensorPointerQueue* q = (TensorPointerQueue*)malloc(sizeof(TensorPointerQueue));
    assert(q);
    q->capacity = initial_capacity > 0 ? initial_capacity : 32; // Default capacity 32
    q->items = (Tensor_float**)malloc(q->capacity * sizeof(Tensor_float*));
    assert(q->items);
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    return q;
}

// Destroys the queue, freeing the internal array.
void tensor_pointer_queue_destroy(TensorPointerQueue* q) 
{
    if (q) 
    {
        free(q->items);
        free(q);
    }
}

// Checks if the queue is empty.
bool tensor_pointer_queue_is_empty(TensorPointerQueue* q) 
{
    return q->count == 0;
}

// Adds a tensor pointer to the back of the queue. Resizes if necessary.
// Uses a simple linear copy on resize, not a proper circular buffer copy for simplicity.
void tensor_pointer_queue_push(TensorPointerQueue* q, Tensor_float* tensor) 
{
    if (!tensor) return; // Don't add NULL
    // Resize if full
    if (q->count >= q->capacity) 
    {
        size_t new_capacity = q->capacity * 2;
        Tensor_float** new_items = (Tensor_float**)malloc(new_capacity * sizeof(Tensor_float*));
        assert(new_items);
        // Simple copy assuming no wrap-around needed (less efficient but simpler)
        if (q->tail > q->head) 
        {
             memcpy(new_items, &q->items[q->head], q->count * sizeof(Tensor_float*));
        } else if (q->count > 0) { // Wrapped around case needs careful copy
             size_t first_part_count = q->capacity - q->head;
             memcpy(new_items, &q->items[q->head], first_part_count * sizeof(Tensor_float*));
             memcpy(&new_items[first_part_count], q->items, q->tail * sizeof(Tensor_float*));
        }
        free(q->items);
        q->items = new_items;
        q->head = 0;
        q->tail = q->count; // Reset tail after linear copy
        q->capacity = new_capacity;
    }
    // Add item at tail
    q->items[q->tail] = tensor;
    q->tail = (q->tail + 1) % q->capacity; // Modulo for basic wrap-around
    q->count++;
}

// Removes and returns the tensor pointer from the front of the queue. Returns NULL if empty.
Tensor_float* tensor_pointer_queue_pop(TensorPointerQueue* q) 
{
    if (q->count == 0) 
    {
        return NULL; // Queue is empty
    }
    Tensor_float* tensor = q->items[q->head];
    q->head = (q->head + 1) % q->capacity; // Modulo for basic wrap-around
    q->count--;
    return tensor;
}
// --- End Queue Emulation ---


// SGD Optimizer struct definition
struct SGD_Optimizer_float 
{
    Optimizer_float base_optimizer; // MUST be first member for casting
    // SGD specific members (e.g., momentum buffer) could be added here if needed
};

// --- Function Prototypes specific to SGD ---
void SGD_Optimizer_float_getParams(Optimizer_float* self, Tensor_float* loss_tensor); // Implementation of base->getParams
void SGD_Optimizer_float_step(Optimizer_float* self); // Implementation of base->step
void SGD_Optimizer_float_destroy(Optimizer_float* self);                   // Implementation of base->destroy
SGD_Optimizer_float* SGD_Optimizer_float_create(float lr);
// Convenience function (optional) - uncommented
void SGD_Optimizer_float_minimise(SGD_Optimizer_float* sgd_optimizer, Tensor_float* loss_tensor);


/*
    Performs a backward traversal (BFS) starting from the `loss_tensor` to find all unique Tensors
    in the computational graph that require gradients. These are considered the parameters
    to be optimized and are added to the optimizer's `params` set.
*/
void SGD_Optimizer_float_getParams(Optimizer_float* self, Tensor_float* loss_tensor) 
{
    assert(self && self->params && loss_tensor && "NULL pointer in SGD getParams");

    tensor_pointer_set_clear(self->params); // Clear out any old params.

    TensorPointerQueue* q = tensor_pointer_queue_create(32);   // Queue for BFS traversal
    TensorPointerSet* visited = tensor_pointer_set_create(32); // Keep track of visited tensors

    tensor_pointer_queue_push(q, loss_tensor);
    tensor_pointer_set_add(visited, loss_tensor);

    while (!tensor_pointer_queue_is_empty(q)) 
    {
        Tensor_float* current_tensor = tensor_pointer_queue_pop(q);

        // If the current tensor itself requires gradients and has no backOp (i.e., it's a leaf/parameter node),
        // add it to the optimizer's parameters.
        // This logic differs from the original comment's description (adding operands).
        // Adding leaves that require_grad seems more standard for optimizers.
        if (current_tensor->requires_grad && current_tensor->backOp == NULL) 
        {
            tensor_pointer_set_add(self->params, current_tensor);
        }

        // Look at the operation that *created* the current tensor
        Operation_float* op = current_tensor->backOp;
        if (op) 
        {
            // Explore the inputs (operands) of the operation
            if (op->t1 != NULL && !tensor_pointer_set_contains(visited, op->t1)) 
            {
                 tensor_pointer_queue_push(q, op->t1);
                 tensor_pointer_set_add(visited, op->t1);
            }
            if (op->t2 != NULL && !tensor_pointer_set_contains(visited, op->t2)) 
            {
                tensor_pointer_queue_push(q, op->t2);
                tensor_pointer_set_add(visited, op->t2);
            }
            // We don't typically add intermediate results (tensors with a backOp) to params,
            // only the learnable leaf nodes (weights, biases).
        }
    }

    // Cleanup traversal helpers
    tensor_pointer_queue_destroy(q);
    tensor_pointer_set_destroy(visited);

    // Optional: Print found parameters for debugging
    // printf("SGD found %zu parameters requiring gradients:\n", self->params->count);
    // for(size_t i=0; i < self->params->count; ++i) {
    //     printf(" - Param %zu: Addr=%p\n", i, (void*)self->params->items[i]);
    // }
}


/*
    Performs one step of Stochastic Gradient Descent for all parameters in `self->params`.
    Updates parameter values using the rule: val = val - learning_rate * grad
*/
void SGD_Optimizer_float_step(Optimizer_float* self) 
{
    assert(self && self->params);
    float learning_rate = self->lr; // Get learning rate from optimizer state

    for (size_t i = 0; i < self->params->count; ++i) 
    {
        Tensor_float* param = self->params->items[i];

        // Ensure the parameter is valid and requires gradient update
        if (param && param->requires_grad && param->val && param->grad) 
        {
            // Verify shapes match before update (should always match if graph is correct)
             assert(matrix_float_verifyShapeForElementwiseOperation(param->val->shape, param->val->num_dims, param->grad->shape, param->grad->num_dims) && "Parameter value and gradient shape mismatch during SGD step!");

             // Calculate: learning_rate * param->grad
             Matrix_float* scaled_grad = matrix_float_multiply_scalar(learning_rate, param->grad);

             // Calculate: param->val - scaled_grad
             Matrix_float* updated_val = matrix_float_subtract(param->val, scaled_grad);

             // Replace old param->val with the updated_val
             matrix_float_destroy(param->val); // Free the old value matrix
             param->val = updated_val;         // Assign the new value matrix (tensor now owns it)

             // Cleanup intermediate matrix
             matrix_float_destroy(scaled_grad);

        } else if (param && param->requires_grad) 
        {
             // Optional: Warn if a parameter requiring grad is missing val or grad
             fprintf(stderr, "Warning: SGD step skipped update for parameter Addr=%p. Missing value or gradient matrix.\n", (void*)param);
        }
         // Ignore parameters that don't require gradients.
    }
}

// Destructor for the SGD Optimizer
void SGD_Optimizer_float_destroy(Optimizer_float* self) 
{
    if (self) 
    {
        optimizer_float_destroy_base(self); // Free base resources (params set)
        free(self); // Free the SGD struct itself
    }
}

// Constructor for the SGD Optimizer
SGD_Optimizer_float* SGD_Optimizer_float_create(float lr) 
{
    SGD_Optimizer_float* sgd = (SGD_Optimizer_float*)malloc(sizeof(SGD_Optimizer_float));
    assert(sgd != NULL);

    // Initialize base optimizer part first
    optimizer_float_init_base((Optimizer_float*)sgd, lr);

    // Set function pointers for SGD specific implementations
    Optimizer_float* base_ptr = (Optimizer_float*)sgd; // Use base pointer for setting
    base_ptr->step = SGD_Optimizer_float_step;
    base_ptr->destroy = SGD_Optimizer_float_destroy;
    base_ptr->getParams = SGD_Optimizer_float_getParams;
    base_ptr->zeroGrad = optimizer_float_default_zeroGrad; // Use the default zeroGrad

    return sgd;
}


/*
    Optional convenience function (commented out in C++ version).
    Combines getParams, step, and zeroGrad.
    This pattern is common but less flexible than calling steps individually.
*/
void SGD_Optimizer_float_minimise(SGD_Optimizer_float* sgd_optimizer, Tensor_float* loss_tensor)
 {
    assert(sgd_optimizer && loss_tensor);
    Optimizer_float* optimizer = (Optimizer_float*)sgd_optimizer; // Use base pointer

    // 1. Get all tensor parameters from the graph ending at loss_tensor
    assert(optimizer->getParams != NULL);
    optimizer->getParams(optimizer, loss_tensor);

    // 2. Update all these parameters via the step function
    assert(optimizer->step != NULL);
    optimizer->step(optimizer); // Pass optimizer state (contains lr)

    // 3. Clear all the gradients of the parameters for the next iteration
    assert(optimizer->zeroGrad != NULL);
    optimizer->zeroGrad(optimizer);
}


// ========================================
// END OF "optims/sgd.h" content
// ========================================


// ========================================
// START OF "data/dataloader.h" content
// ========================================
/*
    This file defines the base struct of each Dataset in the project. The data is stored in a simple
    dynamic array and each object in the array is a pair signifying input and target (ground truth).
*/

// Base DataLoader struct (conceptual base)
struct DataLoader_float_float 
{
    // Dynamic array of input/target pairs
    DataLoader_Pair_float_float* data;
    size_t num_data;      // Current number of data points stored
    size_t capacity_data; // Allocated capacity of the data array

    // --- Function pointers for polymorphism (emulating virtual methods) ---
    // Adds a single data pair (input, target) to the loader.
    void (*add)(DataLoader_float_float* self, float input, float target);
    // Populates the loader with a specified number of examples (logic specific to derived loader).
    void (*create_examples)(DataLoader_float_float* self, int num_examples);
    // Destroys the data loader, freeing allocated memory.
    void (*destroy)(DataLoader_float_float* self);
};

// Base initializer (called by specific loader create functions)
void dataloader_float_float_init_base(DataLoader_float_float* self, size_t initial_capacity) 
{
    self->capacity_data = initial_capacity > 0 ? initial_capacity : 10; // Default capacity 10
    self->data = (DataLoader_Pair_float_float*)malloc(self->capacity_data * sizeof(DataLoader_Pair_float_float));
    assert(self->data);
    self->num_data = 0;
    // Pointers must be set by derived type's create function
    self->add = NULL;
    self->create_examples = NULL;
    self->destroy = NULL;
}

// Base destructor helper (called by specific loader destroy functions)
void dataloader_float_float_destroy_base(DataLoader_float_float* self) 
{
    if (self) 
    {
        free(self->data); // Free the data array
        // Don't free 'self' here, derived destroyer does that.
    }
}

// Generic add function (can be used by specific implementations). Handles resizing.
void dataloader_float_float_base_add(DataLoader_float_float* self, float input, float target) 
{
    // Resize if necessary
    if (self->num_data >= self->capacity_data) 
    {
        self->capacity_data *= 2;
        self->data = (DataLoader_Pair_float_float*)realloc(self->data, self->capacity_data * sizeof(DataLoader_Pair_float_float));
        assert(self->data);
    }
    // Add the data pair
    self->data[self->num_data].input = input;
    self->data[self->num_data].target = target;
    self->num_data++;
}
// ========================================
// END OF "data/dataloader.h" content
// ========================================


// ========================================
// START OF "data/celsius2fahrenheit.h" content
// ========================================
/*
    This file defines the Celsius To Fahrenheit DataLoader. It generates pairs of (celsius, fahrenheit)
    values for training simple linear models.

    Usage Example:
    ```c
    // Seed random numbers once at the start of your program
    srand(time(NULL));

    Celsius2Fahrenheit_DataLoader_float_float* loader = Celsius2Fahrenheit_DataLoader_float_float_create();
    loader->create_examples((DataLoader_float_float*)loader, 50); // Create 50 examples

    printf("Generated %zu examples:\n", loader->base_loader.num_data);
    for(size_t i = 0; i < loader->base_loader.num_data; ++i) 
    {
        float celsius_in = loader->base_loader.data[i].input;
        float fahrenheit_target = loader->base_loader.data[i].target;
        printf("  Input: %.1f C, Target: %.1f F\n", celsius_in, fahrenheit_target);
        // Use this data for training...
    }

    loader->base_loader.destroy((DataLoader_float_float*)loader); // Destroy the loader
    ```
*/

// Specific DataLoader struct for Celsius to Fahrenheit conversion
struct Celsius2Fahrenheit_DataLoader_float_float 
{
    DataLoader_float_float base_loader; // MUST be first member for casting
    int MAX_CELSIUS; // Upper bound (exclusive) for random celsius generation
};

// Helper function to convert celsius to fahrenheit
float celsius_to_fahrenheit(float input) 
{
    return (9.0f * input / 5.0f) + 32.0f;
}

// --- Implementation of base function pointers ---

// Adds a training example using the base resizing logic.
void Celsius2Fahrenheit_DataLoader_float_float_add(DataLoader_float_float* base_self, float input, float target) {
    // This specific implementation just forwards to the base add function.
    dataloader_float_float_base_add(base_self, input, target);
}

// Populates the dataset with randomly generated celsius values and their fahrenheit equivalents.
void Celsius2Fahrenheit_DataLoader_float_float_create_examples(DataLoader_float_float* base_self, int num_examples) 
{
    // Cast base pointer to specific type to access MAX_CELSIUS
    Celsius2Fahrenheit_DataLoader_float_float* self = (Celsius2Fahrenheit_DataLoader_float_float*)base_self;
    assert(self && num_examples >= 0);
    // Seed random numbers if not done elsewhere (better to seed in main)
    // srand(time(NULL));

    for (int i = 0; i < num_examples; i++) 
    {
        // Generate random float celsius value between 0 and MAX_CELSIUS (approx)
        // Or use integer celsius: float input = (float)(rand() % self->MAX_CELSIUS);
        float input = (float)rand() / (float)(RAND_MAX / self->MAX_CELSIUS); // random float value between 0 and MAX_CELSIUS
        float target = celsius_to_fahrenheit(input);

        // Call the add function via the function pointer in the base struct
        // Pass the base_self pointer, which add function expects.
        self->base_loader.add(base_self, input, target);
    }
}

// Destructor for the Celsius2Fahrenheit DataLoader
void Celsius2Fahrenheit_DataLoader_float_float_destroy(DataLoader_float_float* base_self) 
{
     if (base_self) 
     {
        dataloader_float_float_destroy_base(base_self); // Free the data array via base helper
        free(base_self); // Free the specific struct itself
     }
}

// Constructor for the Celsius2Fahrenheit DataLoader
Celsius2Fahrenheit_DataLoader_float_float* Celsius2Fahrenheit_DataLoader_float_float_create() 
{
     Celsius2Fahrenheit_DataLoader_float_float* loader = (Celsius2Fahrenheit_DataLoader_float_float*)malloc(sizeof(Celsius2Fahrenheit_DataLoader_float_float));
     assert(loader);

     // Initialize base part first
     dataloader_float_float_init_base((DataLoader_float_float*)loader, 10); // Initial capacity 10

     // Set specific members
     loader->MAX_CELSIUS = 100; // Set default max celsius

     // Set function pointers in the base struct to point to the specific implementations
     loader->base_loader.add = Celsius2Fahrenheit_DataLoader_float_float_add;
     loader->base_loader.create_examples = Celsius2Fahrenheit_DataLoader_float_float_create_examples;
     loader->base_loader.destroy = Celsius2Fahrenheit_DataLoader_float_float_destroy;

     return loader;
}
// ========================================
// END OF "data/celsius2fahrenheit.h" content
// ========================================


// ========================================
// START OF "dense.h" content
// ========================================
/*
    Defines a standard fully connected (dense) layer for a neural network.
*/

// Activation function types
typedef enum 
{
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU, // Note: RELU forward/backward operations not implemented above
    ACTIVATION_NONE // Linear activation
} activation_enum;

// Weight initialization schemes
typedef enum 
{
    // CHANGED BACK: Renamed to match usage in provided compiler errors
    INITIALISATION_GLOROT,
    INITIALISATION_ZEROS // Example: For biases or specific cases
    // Add others like He initialization if needed
} initialisation_enum;


/*
    Dense Layer Structure
*/
struct Dense_float 
{
    // Weights are stored as a Tensor [input_size, output_size]. Owned by the layer.
    Tensor_float* weights;

    // Biases are stored as a Tensor [1, output_size]. Owned by the layer.
    Tensor_float* biases;

    // Layer dimensions
    int input_size;
    int output_size;

    // Activation function type for this layer
    activation_enum act;

    // Optional: Store initialization types used (for info)
    initialisation_enum weight_init;
    initialisation_enum bias_init;

};


// --- Dense Layer Function Prototypes ---
// Helper to initialize weight/bias data based on enum
float* dense_float_initData(int fan_in, int fan_out, initialisation_enum init);

Dense_float* dense_float_create(int input_size, int output_size, activation_enum act, initialisation_enum weight_init, initialisation_enum bias_init);
void dense_float_destroy(Dense_float* layer);
Tensor_float* dense_float_forward(Dense_float* self, Tensor_float* x);
// Removed updateGradients, as optimizer handles updates.


// --- Implementations ---

// Helper to initialize weight or bias data array
float* dense_float_initData(int fan_in, int fan_out, initialisation_enum init) 
{
    size_t num_elements = (size_t)fan_in * fan_out; // fan_in will be 1 for bias
    float* data = NULL;

    switch (init) 
    {
        case INITIALISATION_GLOROT: // CHANGED BACK
            // Glorot uses fan_in and fan_out of the weight matrix
            data = utils_glorotInit_float(fan_in, fan_out);
            break;
        case INITIALISATION_ZEROS:
            data = (float*)calloc(num_elements, sizeof(float));
            assert(data != NULL);
            break;
        // Add cases for other initializations here (e.g., He)
        default:
            fprintf(stderr, "Error: Unknown initialisation type requested (%d).\n", init);
            // Return zeros as a fallback
            data = (float*)calloc(num_elements, sizeof(float));
            assert(data != NULL);
            break;
    }
    return data; // Caller must free if create doesn't copy (it does)
}

/*
    Constructor for a Dense layer. Initializes weights and biases according to specified schemes.
    Weights and Biases are created as Tensors that require gradients, making them learnable parameters.
    NOTE: This function requires 5 arguments (input_size, output_size, act, weight_init, bias_init).
          Calls to this function in older code (like main_c_tests.c) must be updated to provide the 5th argument (bias_init).
*/
Dense_float* dense_float_create(int input_size, int output_size, activation_enum act, initialisation_enum weight_init, initialisation_enum bias_init) 
{
    assert(input_size > 0 && output_size > 0);

    Dense_float* layer = (Dense_float*)malloc(sizeof(Dense_float));
    assert(layer != NULL);

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->act = act;
    layer->weight_init = weight_init;
    layer->bias_init = bias_init;

    // Initialize weights Tensor [input_size, output_size] (Requires Grad = True)
    int weightShape[] = {input_size, output_size};
    float* weightVal = dense_float_initData(input_size, output_size, weight_init);
    layer->weights = tensor_float_create_from_values(weightVal, weightShape, 2, true); // true = requires_grad
    free(weightVal); // create copies the data

    // Initialize biases Tensor [1, output_size] (Requires Grad = True)
    int biasShape[] = {1, output_size};
    float* biasVal = dense_float_initData(1, output_size, bias_init); // fan_in=1 for bias init data generation if needed
    layer->biases = tensor_float_create_from_values(biasVal, biasShape, 2, true); // true = requires_grad
    free(biasVal); // create copies the data

    return layer;
}

// Destructor for Dense layer (frees owned tensors)
void dense_float_destroy(Dense_float* layer) 
{
    if (layer) 
    {
        // Destroy the tensors owned by the layer
        tensor_float_destroy(layer->weights);
        tensor_float_destroy(layer->biases);
        // Free the layer struct itself
        free(layer);
    }
}


/*
    Performs the forward pass of the dense layer: activation(input @ weights + biases).

    - Input `x` is expected to have shape [batch_size, input_size].
    - Weights `W` have shape [input_size, output_size].
    - Biases `b` have shape [1, output_size].
    - Output `y` has shape [batch_size, output_size].

    Broadcasting Note: The addition `(x @ W) + b` requires broadcasting the bias `b`
    across the batch dimension of the dot product result. The current `tensorOps_add_float`
    does NOT support broadcasting and requires identical shapes.

    Workaround: If `tensorOps_add_float` is not modified to support broadcasting,
    this forward pass will only work correctly if the input batch size is 1,
    or if bias addition is manually implemented with broadcasting logic, or skipped.

    Let's assume for now that a broadcasting add (`tensorOps_add_broadcast_float`) exists
    or that the user handles batching appropriately outside. We will *call* the standard add
    but acknowledge its limitation.
*/
Tensor_float* dense_float_forward(Dense_float* self, Tensor_float* x) 
{
    assert(self && x && self->weights && self->biases);
    assert(x->val && x->val->num_dims >= 1); // Allow 1D input (e.g. single example)
    assert(x->val->shape[x->val->num_dims - 1] == self->input_size && "Input tensor last dimension incompatible with dense layer input size");

    // --- Step 1: Calculate dot product: z = x @ weights ---
    // x shape: [batch, input_size] or [input_size] -> promote to [1, input_size]?
    // W shape: [input_size, output_size]
    // z shape: [batch, output_size]
    Tensor_float* z = tensorOps_dot_float(x, self->weights);
    assert(z && z->val); // Ensure dot product succeeded

    // --- Step 2: Add biases: z_biased = z + biases ---
    // z shape: [batch, output_size]
    // b shape: [1, output_size]
    // Requires broadcasting 'b' over the batch dimension of 'z'.
    Tensor_float* z_biased;
    // Check if shapes match exactly (batch size 1) or if bias shape is broadcastable (row vector)
    if (matrix_float_verifyShapeForElementwiseOperation(z->val->shape, z->val->num_dims, self->biases->val->shape, self->biases->val->num_dims)) 
    {
        // Shapes match exactly (e.g., batch size 1) - use standard add
        z_biased = tensorOps_add_float(z, self->biases);
    }
    else if (z->val->num_dims == 2 && self->biases->val->num_dims == 2 && self->biases->val->shape[0] == 1 && z->val->shape[1] == self->biases->val->shape[1]) 
    {
        // Broadcast scenario: z=[B, out], b=[1, out]
        // TODO: Implement tensorOps_add_broadcast_float or manual broadcast add here.
        fprintf(stderr, "Warning: Dense layer bias addition requires broadcasting, which is not fully implemented in tensorOps_add_float. Using standard add (likely incorrect for batch > 1).\n");
        // Attempting standard add - WILL LIKELY FAIL ASSERT in matrix_add if B > 1
         z_biased = tensorOps_add_float(z, self->biases); // Placeholder - Needs fix
    }
    else 
    {
         fprintf(stderr, "Error: Shape mismatch in dense layer bias addition that cannot be broadcasted (z shape [%d, %d] vs bias shape [%d, %d]).\n", z->val->shape[0], z->val->shape[1], self->biases->val->shape[0], self->biases->val->shape[1] );
         // Simplified check for 2D cases, enhance if needed
         if (z->val->num_dims == 2 && self->biases->val->num_dims == 2) 
         {
            fprintf(stderr, "Error: Shape mismatch in dense layer bias addition that cannot be broadcasted (z shape [%d, %d] vs bias shape [%d, %d]).\n", z->val->shape[0], z->val->shape[1], self->biases->val->shape[0], self->biases->val->shape[1] );
         } else 
         {
            fprintf(stderr, "Error: Shape mismatch in dense layer bias addition involving non-2D tensors (broadcasting not supported).\n");
         }

         assert(false && "Unsupported shapes for bias addition.");
         z_biased = z; // Should not reach here
    }
    // IMPORTANT: 'z' is now an intermediate tensor in the graph, referenced by z_biased->backOp->t1. Do not destroy 'z'.

    // --- Step 3: Apply activation function: logits = activation(z_biased) ---
    Tensor_float* logits = NULL;
    switch (self->act) 
    {
        case ACTIVATION_SIGMOID:
            logits = tensorOps_sigmoid_float(z_biased);
            break;
        case ACTIVATION_RELU:
            // logits = tensorOps_relu_float(z_biased); // Assuming tensorOps_relu_float exists
            fprintf(stderr, "Error: RELU activation not implemented in tensorOps.\n");
            assert(false && "RELU activation missing.");
            logits = z_biased; // Fallback - Incorrect
            break;
        case ACTIVATION_NONE:
            logits = z_biased; // No activation, pass z_biased through
            // Note: If no activation, z_biased IS the result. We don't need a new tensor.
            // The ownership chain remains: logits points to z_biased.
            break;
        default:
             fprintf(stderr, "Error: Unknown activation type (%d) in dense layer.\n", self->act);
             assert(false && "Unknown activation.");
             logits = z_biased; // Fallback - Incorrect
             break;
    }
    // IMPORTANT: 'z_biased' is now an intermediate tensor, referenced by logits->backOp->t1 (if activation op was applied). Do not destroy 'z_biased'.

    // Memory Management Summary for Forward Pass:
    // - Input 'x' is not destroyed.
    // - Intermediate 'z' (result of dot) is owned by the bias add operation.
    // - Intermediate 'z_biased' (result of bias add) is owned by the activation operation (or is the final result if activation is NONE).
    // - The final 'logits' tensor is returned. It's owned by the last operation in the chain (activation or bias add).
    // - Graph cleanup must eventually destroy the operations, which in turn destroy their result tensors.

    return logits;
}

/*
    Function removed: dense_float_updateGradients
    Reason: Parameter updates are handled by the optimizer (e.g., SGD_Optimizer_float_step),
    which iterates through the parameters collected by `getParams` and applies the update rule.
    The layer itself doesn't need an update function.
*/
// void dense_float_updateGradients(Dense_float* self, float lr) 
// {
//     UNUSED(self); // Mark parameters as unused
//     UNUSED(lr);
//     // This function is unnecessary when using an Optimizer.
// };

// ========================================
// END OF "dense.h" content
// ========================================


#endif // BUILD_TENSORFLOW_FLOAT_H


/*
--- END OF REVISED FILE buildTensorflow_float1.h ---

--- END OF FILE buildTensorflow_float2.h ---
```
*/

