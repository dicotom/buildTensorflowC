#include "buildTensorflow_float3.h" // Use the provided header name
#include <stdio.h>                  // For printf
#include <stdlib.h>                 // For exit, EXIT_FAILURE if asserts fail hard
#include <time.h>                   // For srand seeding

// Example of training a network on the buildTensorflow framework (C version).
int main() 
{

    // Seed the random number generator (important for initializations like Glorot)
    srand(time(NULL));

    // Load Dataset
    // The create function populates the ->data member
    Celsius2Fahrenheit_DataLoader_float_float* dataset = Celsius2Fahrenheit_DataLoader_float_float_create();
    if (!dataset) 
    {
        fprintf(stderr, "Failed to create dataset loader.\n");
        return EXIT_FAILURE;
    }
    // Use the function pointer from the base struct to call create_examples
    // Note: The C++ code used 5 examples.
    printf("Creating dataset examples...\n");
    dataset->base_loader.create_examples((DataLoader_float_float*)dataset, 50); // Create 50 examples as per header comment example (or 5 to match C++)? Let's use 50 for now.
    printf("Dataset created with %zu examples.\n", dataset->base_loader.num_data);


    // Create Model
    // dense_float_create requires input_size, output_size, activation, weight_init, bias_init
    // Using Glorot for weights and Zeros for bias as common defaults.
    printf("Creating model...\n");
    Dense_float* fc1 = dense_float_create(1, 1, ACTIVATION_NONE, INITIALISATION_GLOROT, INITIALISATION_ZEROS);
     if (!fc1) 
     {
        fprintf(stderr, "Failed to create dense layer.\n");
        // Cleanup previously allocated memory
        dataset->base_loader.destroy((DataLoader_float_float*)dataset);
        return EXIT_FAILURE;
     }

    // Initialise Optimiser
    printf("Creating optimizer...\n");
    SGD_Optimizer_float* sgd = SGD_Optimizer_float_create(0.01f); // Use 0.01f for float literal
     if (!sgd) 
    {
        fprintf(stderr, "Failed to create SGD optimizer.\n");
        // Cleanup previously allocated memory
        dense_float_destroy(fc1);
        dataset->base_loader.destroy((DataLoader_float_float*)dataset);
        return EXIT_FAILURE;
    }

    // Train
    printf("Training started\n");
    int num_epochs = 2000;
    for (int j = 0; j < num_epochs; j++) 
    {
        // Basic loss tracking for the epoch
        float epoch_loss_sum = 0.0f;
        size_t epoch_batches = 0;

        // Iterate through the dataset using standard C loop
        for (size_t i = 0; i < dataset->base_loader.num_data; i++) 
        {
            // Get data for this iteration
            float celsius_val = dataset->base_loader.data[i].input;
            float fahrenheit_val = dataset->base_loader.data[i].target;

            // Create Tensors for input and target
            // Input data and target data usually don't require gradients themselves.
            float inp_data[] = {celsius_val};
            int inp_shape[] = {1, 1};
            Tensor_float* inp = tensor_float_create_from_values(inp_data, inp_shape, 2, false);

            float tar_data[] = {fahrenheit_val};
            int tar_shape[] = {1, 1};
            Tensor_float* tar = tensor_float_create_from_values(tar_data, tar_shape, 2, false);

            // --- Forward Prop ---
            // The forward pass builds the computation graph implicitly
            Tensor_float* out = dense_float_forward(fc1, inp); // out = fc1(inp) = W*inp + b

            // --- Get Loss (Squared Error: (out - tar)^2) ---
            // Need a tensor for -1 to calculate -tar
            float l_data[] = {-1.0f};
            int l_shape[] = {1, 1};
            Tensor_float* l = tensor_float_create_from_values(l_data, l_shape, 2, false);

            // k = l * tar = (-1) * tar = -tar
            Tensor_float* k = tensorOps_multiply_float(l, tar);

            // loss_diff = out + k = out - tar
            Tensor_float* loss_diff = tensorOps_add_float(out, k);

            // finalLoss = loss_diff ^ 2 = (out - tar)^2
            Tensor_float* finalLoss = tensorOps_power_float(loss_diff, 2.0f);

             // Check if finalLoss or its value is NULL before accessing
            if (finalLoss && finalLoss->val && finalLoss->val->val) 
            {
                 epoch_loss_sum += finalLoss->val->val[0]; // Accumulate scalar loss value
                 epoch_batches++;
            } else 
            {
                fprintf(stderr, "Warning: finalLoss tensor invalid during training step %zu of epoch %d.\n", i, j);
            }

            // --- Compute backProp ---
            // Start backpropagation from the final loss tensor.
            // This calculates gradients for all tensors in the graph that require them (weights, biases).
            tensor_float_backward_default(finalLoss);

            // --- Perform Gradient Descent ---
            // The minimise function finds parameters connected to finalLoss,
            // updates them using their calculated gradients and the learning rate,
            // and zeros the gradients for the next iteration.
            SGD_Optimizer_float_minimise(sgd, finalLoss);

            // --- Cleanup Tensors created in this loop iteration ---
            // It's crucial to destroy tensors created *within* the loop.
            // NOTE: This cleanup might be incomplete. Intermediate tensors (out, k, loss_diff)
            // created by operations are owned by those operations. If the framework doesn't
            // automatically garbage collect or manage the operation nodes created during the forward pass,
            // those intermediate tensors and operations might leak memory. Destroying finalLoss
            // DOES NOT automatically destroy the operations/tensors that led to it based on the header analysis.
            // We destroy what we explicitly created with _create_from_values and the final loss node.
            tensor_float_destroy(inp);
            tensor_float_destroy(tar);
            tensor_float_destroy(l);
            tensor_float_destroy(finalLoss);
             // Intermediate tensors k and loss_diff are implicitly managed (or leaked)
             // Tensor 'out' is also intermediate (output of dense_float_forward)
        } // End of dataset loop

        // Optional: Print epoch loss
         if ((j + 1) % 100 == 0 && epoch_batches > 0) 
         { // Print every 100 epochs
            printf("Epoch [%d/%d], Average Loss: %.6f\n", j + 1, num_epochs, epoch_loss_sum / epoch_batches);
         }

    } // End of epoch loop

    printf("Training completed\n");

    // --- Inference ---
    float cel = 4.0f; // Test value
    float test_data[] = {cel};
    int test_shape[] = {1, 1};
    Tensor_float* test = tensor_float_create_from_values(test_data, test_shape, 2, false);
     if (!test) 
     {
         fprintf(stderr, "Failed to create test tensor.\n");
          // Cleanup allocated memory before exiting
         sgd->base_optimizer.destroy((Optimizer_float*)sgd);
         dense_float_destroy(fc1);
         dataset->base_loader.destroy((DataLoader_float_float*)dataset);
         return EXIT_FAILURE;
     }


    printf("Performing inference for %.1f C...\n", cel);
    Tensor_float* out1 = dense_float_forward(fc1, test);

    if (out1 && out1->val && out1->val->val) 
    {
        printf("The conversion of %.1f degrees celcius to fahrenheit is %.4f\n", cel, out1->val->val[0]);
        // Expected output for 4 C should be near 39.2 F
    } else 
    {
        printf("Inference failed or produced invalid tensor.\n");
    }

    // --- Clean up ---
    // Destroy inference tensors
    tensor_float_destroy(out1);
    tensor_float_destroy(test);

    // Destroy model, optimizer, and dataset
    dense_float_destroy(fc1);
    // Call destroy via the function pointer in the base struct
    sgd->base_optimizer.destroy((Optimizer_float*)sgd);
    dataset->base_loader.destroy((DataLoader_float_float*)dataset);

    printf("Cleanup finished.\n");

    return 0;
}