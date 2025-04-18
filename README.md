This is the porting to c of karanchahal/buildTensorflow https://github.com/karanchahal/buildTensorflow/tree/develop

To Compile:
tests.c
gcc tests.c -o run_c_tests -lm -Wall -Wextra -pedantic -std=c99

-lm: Links the math library (for expf, powf, fabsf).
-Wall -Wextra -pedantic -std=c99: Recommended flags for catching potential issues.


main.c
gcc -Wall -Wextra -o main main.c -lm

gcc: Invokes the GCC compiler. (Use clang if you prefer Clang).
-Wall -Wextra: Enables most standard and extra compiler warnings. Highly recommended to catch potential issues.
-o main: Specifies the name of the output executable file (you can call it something else if you like).
main.c: The name of your C source file.
-lm: Crucial! This links the standard math library (libm). Your code uses functions like sqrt, expf, powf (defined in <math.h> which is included by your header), and these require explicitly linking the math library on many systems (especially Linux).


Okay, let's break down how to use this C library for building and training simple neural networks, based on the provided buildTensorflow_float3.h header and the usage examples in tests.c.

This library provides functionality for:

N-Dimensional Matrices (Matrix_float): Basic data storage.

Tensors (Tensor_float): Wrappers around matrices that track operations for automatic differentiation (autograd).

Operations (tensorOps_...): Functions that build a computational graph using Tensors.

Layers (Dense_float): Pre-built neural network layers.

Optimizers (SGD_Optimizer_float): Algorithms to update model parameters based on gradients.

Data Loaders (Celsius2Fahrenheit_DataLoader_float_float): Helpers to generate or load training data.

Tutorial: Using the buildTensorflow_float C Library
1. Setup

First, include the library header and standard C libraries you might need:

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h> // For srand

// Include the library header
#include "buildTensorflow_float3.h" // Make sure this path is correct

// Define a small value for floating-point comparisons if needed
#define FLOAT_EPSILON 1e-5

int main() {
    // Seed random number generator (important for weight initialization)
    srand(time(NULL));

    printf("Starting Tutorial...\n");

    // --- Tutorial examples go here ---

    printf("Tutorial Finished.\n");
    return 0;
}
```


2. Matrices (Matrix_float)

Matrices are the basic data containers. They store N-dimensional arrays of floats.

Creating a Matrix:

You need the data (as a float array) and the shape (as an int array). The total number of elements in the data array must match the product of the dimensions in the shape array.


```c
// Example: Create a 2x3 matrix

float matrix_data[] = {1.0f, 2.0f, 3.0f,4.0f, 5.0f, 6.0f}; // 6 elements

int matrix_shape[] = {2, 3}; // Shape: 2 rows, 3 columns

int matrix_dims = 2; // Number of dimensions

Matrix_float* mat_a = matrix_float_create(matrix_data, matrix_shape, matrix_dims);
if (!mat_a) 
{
     fprintf(stderr, "Failed to create matrix A\n");
     return 1; // Or handle error appropriately
}

// Example: Create another 2x3 matrix
float matrix_data_b[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
Matrix_float* mat_b = matrix_float_create(matrix_data_b, matrix_shape, matrix_dims);
if (!mat_b) 
{
 fprintf(stderr, "Failed to create matrix B\n");
 matrix_float_destroy(mat_a); // Clean up previously created matrix
 return 1;
}
```

Printing a Matrix:

Use matrix_float_print to see the contents and shape.

```c
printf("Matrix A:\n");
matrix_float_print(mat_a, stdout);

printf("\nMatrix B:\n");
matrix_float_print(mat_b, stdout);
```

The equivalent in tensorflow would be:

```c
import tensorflow as tf

# Define the data and shape
matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
matrix_shape = [2, 3]
matrix_dims = len(matrix_shape)  # == 2

# Create the tensor (matrix)
mat_a = tf.constant(matrix_data, shape=matrix_shape, dtype=tf.float32)

print(mat_a)
```
Output:

```c
tf.Tensor(
[[1. 2. 3.]
[4. 5. 6.]], shape=(2, 3), dtype=float32)
```

















Matrix Operations:

The library provides functions for element-wise operations (+, -, *, /), dot products, and scalar operations.

    // Element-wise addition: mat_c = mat_a + mat_b
    Matrix_float* mat_c = matrix_float_add(mat_a, mat_b);
    printf("\nMatrix C (A + B):\n");
    matrix_float_print(mat_c, stdout);

    // Scalar multiplication: mat_d = 2.0 * mat_a
    Matrix_float* mat_d = matrix_float_multiply_scalar(2.0f, mat_a);
    printf("\nMatrix D (2.0 * A):\n");
    matrix_float_print(mat_d, stdout);

    // Dot product example: Requires compatible shapes
    // Let's create mat_e (3x2) for mat_a (2x3) . mat_e (3x2) -> result (2x2)
    float matrix_data_e[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int matrix_shape_e[] = {3, 2};
    Matrix_float* mat_e = matrix_float_create(matrix_data_e, matrix_shape_e, 2);

    Matrix_float* mat_f = matrix_float_dot(mat_a, mat_e);
    printf("\nMatrix F (A . E):\n");
    matrix_float_print(mat_f, stdout);


Destroying Matrices:

CRUCIAL: You must manually free the memory allocated for each matrix when you are done with it using matrix_float_destroy.

// Clean up all created matrices
    matrix_float_destroy(mat_a);
    matrix_float_destroy(mat_b);
    matrix_float_destroy(mat_c); // Result of add operation
    matrix_float_destroy(mat_d); // Result of scalar multiply
    matrix_float_destroy(mat_e);
    matrix_float_destroy(mat_f); // Result of dot product


3. Tensors (Tensor_float)

Tensors are the core components for automatic differentiation. They wrap a Matrix_float (val) and can optionally store another Matrix_float for gradients (grad). They also track the operations performed on them (backOp, frontOp).

Creating Tensors:

You typically create tensors from data and shape, specifying whether they require gradient calculation (requires_grad). Tensors that represent model parameters (weights, biases) or inputs that need gradients must have requires_grad = true. Input data or fixed values usually have requires_grad = false.

    // Create a tensor that requires gradients (e.g., a weight)
    float weight_data[] = {0.5f};
    int scalar_shape[] = {1};
    Tensor_float* w = tensor_float_create_from_values(weight_data, scalar_shape, 1, true); // Requires grad

    // Create a tensor for input data (doesn't require grad itself)
    float input_data[] = {2.0f};
    Tensor_float* x = tensor_float_create_from_values(input_data, scalar_shape, 1, false); // Does NOT require grad

    // Create a bias tensor (requires grad)
    float bias_data[] = {1.0f};
    Tensor_float* b = tensor_float_create_from_values(bias_data, scalar_shape, 1, true); // Requires grad

    printf("\nInitial Tensors:\n");
    printf("w (val): "); matrix_float_print(w->val, stdout);
    printf("w (grad): "); w->grad ? matrix_float_print(w->grad, stdout) : printf("NULL\n"); // Grad is initialized if requires_grad=true
    printf("x (val): "); matrix_float_print(x->val, stdout);
    printf("b (val): "); matrix_float_print(b->val, stdout);


Tensor Operations & Computational Graph:

Use the tensorOps_... functions to perform operations. These functions automatically create Operation_float objects behind the scenes and link the tensors, forming a computational graph.

    // Build a simple computation: y = w * x + b

    // Step 1: a = w * x
    Tensor_float* a = tensorOps_multiply_float(w, x);
    // 'a' now has a->backOp pointing to the MultiplyOperation
    // 'w' and 'x' now have w->frontOp and x->frontOp pointing to the MultiplyOperation

    // Step 2: y = a + b
    Tensor_float* y = tensorOps_add_float(a, b);
    // 'y' now has y->backOp pointing to the AddOperation
    // 'a' and 'b' now have a->frontOp and b->frontOp pointing to the AddOperation

    printf("\nCalculation Result (y->val):\n");
    matrix_float_print(y->val, stdout); // Should be (0.5 * 2.0) + 1.0 = 2.0


Automatic Differentiation (Backward Pass):

To calculate gradients, call tensor_float_backward_default(output_tensor). This starts the backpropagation process from the output_tensor (usually the loss), assuming the gradient of the output with respect to itself is 1. Gradients are accumulated in the .grad matrix of tensors where requires_grad is true.

    // Calculate gradients (d(y)/dw, d(y)/db, d(y)/dx)
    // Since y = w*x + b:
    // dy/da = 1, dy/db = 1
    // da/dw = x, da/dx = w
    // dy/dw = dy/da * da/dw = 1 * x = x = 2.0
    // dy/db = 1
    // dy/dx = dy/da * da/dx = 1 * w = w = 0.5 (but x doesn't require grad)
    tensor_float_backward_default(y);

    printf("\nGradients after backward pass:\n");
    printf("w->grad: "); w->grad ? matrix_float_print(w->grad, stdout) : printf("NULL\n"); // Expected: [2.0]
    printf("b->grad: "); b->grad ? matrix_float_print(b->grad, stdout) : printf("NULL\n"); // Expected: [1.0]
    printf("x->grad: "); x->grad ? matrix_float_print(x->grad, stdout) : printf("NULL\n"); // Expected: NULL (requires_grad=false)
    printf("a->grad: "); a->grad ? matrix_float_print(a->grad, stdout) : printf("NULL\n"); // Grad computed but maybe not needed if intermediate. Expected: [1.0]
    printf("y->grad: "); y->grad ? matrix_float_print(y->grad, stdout) : printf("NULL\n"); // Grad computed. Expected: [1.0]


Zeroing Gradients:

Before the next forward/backward pass in a training loop, you must zero out the gradients from the previous step. This is typically done using the optimizer's zeroGrad method (see Section 5), which acts on the parameters it manages. You can also manually call tensor_float_zeroGrad(tensor).

    // Example of manual zeroing (usually optimizer handles this)
    tensor_float_zeroGrad(w);
    tensor_float_zeroGrad(b);
    printf("\nGradients after zeroGrad:\n");
    printf("w->grad: "); w->grad ? matrix_float_print(w->grad, stdout) : printf("NULL\n");
    printf("b->grad: "); b->grad ? matrix_float_print(b->grad, stdout) : printf("NULL\n");


Destroying Tensors & Graph Issues:

CRUCIAL: Destroy tensors using tensor_float_destroy. HOWEVER, the current library structure has memory management challenges. Destroying a tensor only frees its val and grad matrices. It does not automatically destroy the operations (backOp, frontOp) or other tensors in the graph.

In simple cases like the y = w*x + b example, you need to destroy all tensors you created manually (w, x, b) and the intermediate tensors (a, y).

This is prone to memory leaks or double frees. A robust solution would require proper graph management (e.g., reference counting or arena allocation), which this library currently lacks. Be very careful with cleanup.

    // Clean up the simple graph example
    // Destroying 'y' does NOT automatically destroy 'a', 'w', 'x', 'b' or the Ops.
    tensor_float_destroy(y);
    tensor_float_destroy(a);
    tensor_float_destroy(w);
    tensor_float_destroy(x);
    tensor_float_destroy(b);
    // Note: The Operation objects created by tensorOps_* are leaked here!



4. Dense Layer (Dense_float)

This represents a standard fully connected layer (output = activation(input @ weights + bias)).

Creating a Dense Layer:

Specify input size, output size, activation function (activation_enum), and initialization methods (initialisation_enum) for weights and biases. The layer automatically creates its weights and biases as Tensors requiring gradients.

    // Create a dense layer: 2 inputs, 5 outputs, Sigmoid activation
    // Use Glorot initialization for weights, Zeros for biases.
    int input_features = 2;
    int output_features = 5;
    Dense_float* dense_layer = dense_float_create(
        input_features,
        output_features,
        ACTIVATION_SIGMOID,       // Activation function
        INITIALISATION_GLOROT,    // Weight initialization
        INITIALISATION_ZEROS      // Bias initialization
    );

    printf("\nDense Layer created:\n");
    printf("  Weights Tensor (requires_grad=%d):\n", dense_layer->weights->requires_grad);
    matrix_float_print(dense_layer->weights->val, stdout);
    printf("  Biases Tensor (requires_grad=%d):\n", dense_layer->biases->requires_grad);
    matrix_float_print(dense_layer->biases->val, stdout);


Forward Pass:

Pass an input tensor through the layer using dense_float_forward. The input tensor should have shape [batch_size, input_size]. The output will have shape [batch_size, output_size].

Broadcasting Note: The current implementation might have issues adding the bias if batch_size > 1, as noted in the header comments. Assume batch size 1 for now or verify broadcasting works.

    // Create a sample input tensor (batch size 1, 2 features)
    float layer_input_data[] = {0.5f, -1.0f};
    int layer_input_shape[] = {1, input_features}; // Batch=1, Features=2
    Tensor_float* layer_input = tensor_float_create_from_values(layer_input_data, layer_input_shape, 2, false); // Input data doesn't need grad

    // Perform forward pass
    Tensor_float* layer_output = dense_float_forward(dense_layer, layer_input);

    printf("\nDense Layer Output (shape %d x %d):\n", layer_output->val->shape[0], layer_output->val->shape[1]);
    matrix_float_print(layer_output->val, stdout);



Destroying a Dense Layer:

This frees the layer struct and the weights and biases Tensors it owns.

    // Clean up layer and its intermediate results/inputs
    tensor_float_destroy(layer_output); // Result of forward pass
    tensor_float_destroy(layer_input);  // Input tensor we created
    dense_float_destroy(dense_layer);   // The layer itself (destroys weights/biases)
    // Note: Ops created during forward pass are leaked!



5. Optimizer (SGD_Optimizer_float)

Optimizers update model parameters based on calculated gradients. SGD (Stochastic Gradient Descent) is provided.

Creating an Optimizer:

Specify the learning rate.

    // Create an SGD optimizer with learning rate 0.01
    float learning_rate = 0.01f;
    SGD_Optimizer_float* sgd = SGD_Optimizer_float_create(learning_rate);
    printf("\nCreated SGD Optimizer with LR = %.4f\n", sgd->base_optimizer.lr);


Using the Optimizer (minimise):

The SGD_Optimizer_float_minimise function is a convenience wrapper used in training loops. Given the final loss tensor of your graph, it performs three steps:

getParams: Traverses the graph backward from the loss tensor to find all Tensors that are parameters (require gradients and are typically leaf nodes like weights/biases). Stores pointers to these parameters internally.

step: Updates the .val matrix of each stored parameter using the rule: param->val = param->val - learning_rate * param->grad.

zeroGrad: Sets the .grad matrix of each stored parameter back to zeros, ready for the next iteration.

NOTE: You must call the backward pass (tensor_float_backward_default(loss)) before calling minimise to ensure the gradients (param->grad) needed by the step function are computed.

    // --- Pretend Training Step ---
    // 1. Assume we have a 'loss' tensor calculated from a model forward pass.
    //    (We'll reuse 'y' from the Tensor section for illustration, assuming it's our loss)
    //    Build y = w*x + b again
    Tensor_float* w_opt = tensor_float_create_from_values((float[]){0.5f}, (int[]){1}, 1, true);
    Tensor_float* x_opt = tensor_float_create_from_values((float[]){2.0f}, (int[]){1}, 1, false);
    Tensor_float* b_opt = tensor_float_create_from_values((float[]){1.0f}, (int[]){1}, 1, true);
    Tensor_float* a_opt = tensorOps_multiply_float(w_opt, x_opt);
    Tensor_float* loss = tensorOps_add_float(a_opt, b_opt); // Our 'loss' tensor

    printf("\nOptimizer Example:\n");
    printf("  Initial w: %.4f, Initial b: %.4f\n", w_opt->val->val[0], b_opt->val->val[0]);

    // 2. Calculate gradients
    tensor_float_backward_default(loss);
    printf("  Gradients - w: %.4f, b: %.4f\n", w_opt->grad->val[0], b_opt->grad->val[0]); // Should be 2.0 and 1.0

    // 3. Perform optimizer step
    SGD_Optimizer_float_minimise(sgd, loss);
    // This finds w_opt, b_opt as params.
    // Updates: w_new = 0.5 - 0.01 * 2.0 = 0.48
    //          b_new = 1.0 - 0.01 * 1.0 = 0.99
    // Zeros grads of w_opt, b_opt.

    printf("  Updated w: %.4f, Updated b: %.4f\n", w_opt->val->val[0], b_opt->val->val[0]);
    printf("  Grads after minimise - w: %.4f, b: %.4f\n", w_opt->grad->val[0], b_opt->grad->val[0]); // Should be 0.0



Destroying an Optimizer:

Frees the optimizer struct and its internal parameter list. Does not destroy the parameter Tensors themselves (they are owned by the model/layer).

    // Clean up optimizer and related tensors
    sgd->base_optimizer.destroy((Optimizer_float*)sgd); // Use the function pointer for destroy
    tensor_float_destroy(loss);
    tensor_float_destroy(a_opt);
    tensor_float_destroy(w_opt);
    tensor_float_destroy(x_opt);
    tensor_float_destroy(b_opt);
    // Ops leaked again!




6. Data Loader (Celsius2Fahrenheit_DataLoader_float_float)

Data loaders help manage training data. The example provided generates Celsius/Fahrenheit pairs.

// Create the data loader
    Celsius2Fahrenheit_DataLoader_float_float* dataset = Celsius2Fahrenheit_DataLoader_float_float_create();

    // Generate some examples
    int num_examples = 10;
    dataset->base_loader.create_examples((DataLoader_float_float*)dataset, num_examples);

    printf("\nGenerated Dataset (%zu examples):\n", dataset->base_loader.num_data);
    for (size_t i = 0; i < dataset->base_loader.num_data; ++i) 
    {
        float celsius = dataset->base_loader.data[i].input;
        float fahrenheit = dataset->base_loader.data[i].target;
        printf("  Example %zu: Input=%.2f C, Target=%.2f F\n", i + 1, celsius, fahrenheit);
    }

    // Destroy the dataset loader
    dataset->base_loader.destroy((DataLoader_float_float*)dataset);



7. Training Loop (Putting it all together)

This example trains a simple linear model (a Dense layer with 1 input, 1 output, no activation) to convert Celsius to Fahrenheit.

    printf("\n--- Training Example: Celsius to Fahrenheit ---\n");

    // 1. Dataset
    Celsius2Fahrenheit_DataLoader_float_float* train_data = Celsius2Fahrenheit_DataLoader_float_float_create();
    train_data->base_loader.create_examples((DataLoader_float_float*)train_data, 50); // 50 training examples

    // 2. Model (1 input, 1 output, no activation)
    Dense_float* model = dense_float_create(1, 1, ACTIVATION_NONE, INITIALISATION_GLOROT, INITIALISATION_ZEROS);
    printf("Initial Model - Weight: %.4f, Bias: %.4f\n", model->weights->val->val[0], model->biases->val->val[0]);

    // 3. Optimizer
    SGD_Optimizer_float* optimizer = SGD_Optimizer_float_create(1e-4f); // Use a small learning rate

    // 4. Training Loop
    int epochs = 5000; // Or more/less depending on convergence
    printf("Starting training for %d epochs...\n", epochs);

    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        float total_epoch_loss = 0.0f;

        for (size_t i = 0; i < train_data->base_loader.num_data; ++i) 
        {
            // --- Get data for this iteration ---
            float input_val[] = {train_data->base_loader.data[i].input};
            float target_val[] = {train_data->base_loader.data[i].target};
            int data_shape[] = {1, 1}; // Batch size 1, feature size 1

            // Create Tensors for input and target (no grad needed for these)
            Tensor_float* inp = tensor_float_create_from_values(input_val, data_shape, 2, false);
            Tensor_float* tar = tensor_float_create_from_values(target_val, data_shape, 2, false);

            // --- Forward Pass ---
            Tensor_float* out = dense_float_forward(model, inp); // model(inp)

            // --- Calculate Loss (MSE: (out - tar)^2) ---
            // error = out - tar = out + (-1.0 * tar)
            Tensor_float* neg_one = tensorOps_create_scalar_tensor_like(-1.0f, tar);
            Tensor_float* neg_tar = tensorOps_multiply_float(neg_one, tar);
            Tensor_float* error = tensorOps_add_float(out, neg_tar);
            // loss = error^2
            Tensor_float* loss_tensor = tensorOps_power_float(error, 2.0f);

            total_epoch_loss += loss_tensor->val->val[0]; // Accumulate loss value

            // --- Backward Pass ---
            tensor_float_backward_default(loss_tensor); // Calculate gradients

            // --- Optimizer Step ---
            SGD_Optimizer_float_minimise(optimizer, loss_tensor); // Update weights/biases, zero gradients

            // --- Cleanup iteration tensors ---
            // Need to destroy intermediates created IN THIS ITERATION
            tensor_float_destroy(inp);
            tensor_float_destroy(tar);
            tensor_float_destroy(neg_one);
            tensor_float_destroy(neg_tar); // op owns -tar? No, op owns error.
            tensor_float_destroy(error);   // op owns loss? No, op owns power result.
            tensor_float_destroy(loss_tensor);
            tensor_float_destroy(out);     // result of dense_forward owned by its last op (bias add or activation)
            // NOTE: The cleanup here is tricky due to the ownership issues. Destroying
            // loss_tensor *should* be the main handle, but the intermediate ops/tensors
            // (like out, error, neg_tar) might be leaked depending on destroy implementation.
            // Be cautious. The provided test code also struggles with this.
        } // End of data iteration loop

        // Print epoch summary
        if ((epoch + 1) % 500 == 0) 
        {
            float avg_loss = total_epoch_loss / train_data->base_loader.num_data;
            printf("  Epoch %d/%d, Avg Loss: %.4f, Weight: %.4f, Bias: %.4f\n",
                   epoch + 1, epochs, avg_loss,
                   model->weights->val->val[0], model->biases->val->val[0]);
        }

    } // End of epoch loop

    printf("Training finished.\n");
    printf("Final Model - Weight: %.4f (Expected ~1.8), Bias: %.4f (Expected ~32.0)\n",
           model->weights->val->val[0], model->biases->val->val[0]);


    // 5. Inference Example
    float test_celsius = 10.0f;
    float expected_fahrenheit = (9.0f * test_celsius / 5.0f) + 32.0f;
    float test_input_data[] = {test_celsius};
    int test_input_shape[] = {1, 1};
    Tensor_float* test_input = tensor_float_create_from_values(test_input_data, test_input_shape, 2, false);
    Tensor_float* predicted_output = dense_float_forward(model, test_input);
    printf("\nInference for %.1f C:\n", test_celsius);
    printf("  Predicted: %.4f F\n", predicted_output->val->val[0]);
    printf("  Expected:  %.4f F\n", expected_fahrenheit);

    // 6. Final Cleanup
    tensor_float_destroy(predicted_output);
    tensor_float_destroy(test_input);
    optimizer->base_optimizer.destroy((Optimizer_float*)optimizer);
    dense_float_destroy(model);
    train_data->base_loader.destroy((DataLoader_float_float*)train_data);
    // Again, many intermediate Ops/Tensors from training likely leaked.



8. Memory Management & Limitations (Recap)

Manual Allocation/Deallocation: You must call the corresponding _destroy function for every object created with a _create function (Matrix_float, Tensor_float, Dense_float, SGD_Optimizer_float, etc.).

Graph Cleanup: The biggest challenge is cleaning up the computational graph. The Tensors and Operations are interconnected. Destroying the final output tensor (e.g., loss_tensor) DOES NOT automatically clean up the intermediate tensors or the operations that created them in this library's current state.

Potential Leaks: Forgetting to destroy objects will lead to memory leaks. The interconnected nature of the graph makes manual cleanup complex and error-prone. In the training loop, intermediate tensors created within the loop need careful handling.

No Broadcasting (Mostly): Matrix and Tensor operations generally expect exact shape matches. Bias addition in the Dense layer is noted as potentially problematic for batch sizes greater than 1.

Simplicity: This is a simplified library, likely for educational purposes. It lacks features of mature frameworks (e.g., robust memory management, GPU support placeholders, extensive layer types, complex optimizers, proper broadcasting).

This tutorial should give you a good starting point for using the basic components of the library. Remember to be extremely careful with memory management.
