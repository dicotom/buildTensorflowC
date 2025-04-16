#include "buildTensorflow_float3.h" // Use the provided header name
#include <stdio.h>                  // For printf
#include <stdlib.h>                 // For exit, EXIT_FAILURE if asserts fail hard
#include <time.h>                   // For srand seeding
#include <math.h>                   // For isnan

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
    printf("Creating dataset examples...\n");
    dataset->base_loader.create_examples((DataLoader_float_float*)dataset, 50); // Create 50 examples
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
    // NOTE: Reduced learning rate significantly to prevent NaN/exploding gradients
    SGD_Optimizer_float* sgd = SGD_Optimizer_float_create(0.0001f); // Use a smaller float literal
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
        bool nan_detected = false; // Flag to stop epoch early if NaN occurs

        // Iterate through the dataset using standard C loop
        for (size_t i = 0; i < dataset->base_loader.num_data; i++)
        {
            // Get data for this iteration
            float celsius_val = dataset->base_loader.data[i].input;
            float fahrenheit_val = dataset->base_loader.data[i].target;

            // Create Tensors for input and target
            // Input data and target data usually don't require gradients themselves.
            float inp_data[] = {celsius_val};
            int inp_shape[] = {1, 1}; // Shape [1, 1] is often expected by layers
            Tensor_float* inp = tensor_float_create_from_values(inp_data, inp_shape, 2, false); // Use 2 dimensions

            float tar_data[] = {fahrenheit_val};
            int tar_shape[] = {1, 1}; // Shape [1, 1]
            Tensor_float* tar = tensor_float_create_from_values(tar_data, tar_shape, 2, false); // Use 2 dimensions

            // Check tensor creation
             if (!inp || !tar) {
                fprintf(stderr, "Failed to create input/target tensors in loop.\n");
                nan_detected = true; // Treat as error
                // Cleanup what we can from this iteration
                tensor_float_destroy(inp); // Safe if NULL
                tensor_float_destroy(tar); // Safe if NULL
                break; // Exit inner loop
             }

            // --- Forward Prop ---
            // The forward pass builds the computation graph implicitly
            Tensor_float* out = dense_float_forward(fc1, inp); // out = fc1(inp) = W*inp + b

            // Check forward pass output
            if (!out || !out->val || !out->val->val || isnan(out->val->val[0])) {
                fprintf(stderr, "Warning: NaN or invalid tensor 'out' detected during forward pass step %zu of epoch %d.\n", i, j);
                tensor_float_destroy(inp);
                tensor_float_destroy(tar);
                tensor_float_destroy(out); // Attempt cleanup
                nan_detected = true;
                break; // Exit inner loop
            }

            // --- Get Loss (Squared Error: (out - tar)^2) ---
            // Need a tensor for -1 to calculate -tar
            float l_data[] = {-1.0f};
            int l_shape[] = {1, 1};
            Tensor_float* l = tensor_float_create_from_values(l_data, l_shape, 2, false);

             if (!l) {
                fprintf(stderr, "Failed to create auxiliary tensor 'l'.\n");
                tensor_float_destroy(inp);
                tensor_float_destroy(tar);
                tensor_float_destroy(out);
                nan_detected = true;
                break;
             }


            // k = l * tar = (-1) * tar = -tar
            Tensor_float* k = tensorOps_multiply_float(l, tar);

            // loss_diff = out + k = out - tar
            Tensor_float* loss_diff = tensorOps_add_float(out, k);

            // finalLoss = loss_diff ^ 2 = (out - tar)^2
            Tensor_float* finalLoss = tensorOps_power_float(loss_diff, 2.0f);

             // Check if finalLoss or its value is NULL or NaN before accessing
            if (finalLoss && finalLoss->val && finalLoss->val->val && !isnan(finalLoss->val->val[0]))
            {
                 epoch_loss_sum += finalLoss->val->val[0]; // Accumulate scalar loss value
                 epoch_batches++;
            } else
            {
                fprintf(stderr, "Warning: finalLoss tensor invalid or NaN during training step %zu of epoch %d.\n", i, j);
                 nan_detected = true;
                 // Cleanup tensors for this iteration before breaking
                 tensor_float_destroy(inp);
                 tensor_float_destroy(tar);
                 tensor_float_destroy(l);
                 tensor_float_destroy(k); // Destroy intermediate if created
                 tensor_float_destroy(loss_diff); // Destroy intermediate if created
                 tensor_float_destroy(finalLoss); // Destroy finalLoss itself
                 // 'out' is intermediate, owned by dense_forward internal ops
                 break; // Exit inner loop
            }

            // --- Compute backProp ---
            // Start backpropagation from the final loss tensor.
            tensor_float_backward_default(finalLoss);

            // --- Perform Gradient Descent ---
            // The minimise function finds parameters, updates them, and zeros grads.
            SGD_Optimizer_float_minimise(sgd, finalLoss); // This finds params, steps, zeros grads

            // --- Cleanup Tensors created EXPLICITLY in this loop iteration ---
            // Intermediate tensors (out, k, loss_diff) created by operations are managed
            // (or leaked) by the framework/operation destructors.
            // We only need to destroy tensors created directly with tensor_float_create*.
            tensor_float_destroy(inp);
            tensor_float_destroy(tar);
            tensor_float_destroy(l);
            // Destroying finalLoss is tricky. SGD_Optimizer_float_minimise might still need it
            // or its graph connections temporarily. If minimise handles the graph traversal
            // and update correctly, the graph structure up to finalLoss should remain valid
            // until the end of minimise. However, the current framework lacks explicit graph
            // management, so we might need to keep finalLoss alive until after minimise?
            // Let's assume minimise doesn't take ownership and we should destroy it.
            // If SGD_minimise *did* destroy the graph nodes it visits, this would be a double free.
            // Given the lack of clear graph ownership, destroying it here is the most consistent
            // approach with destroying other explicitly created tensors, but it's risky.
            tensor_float_destroy(finalLoss);
            // tensor_float_destroy(k); // Owned by multiply op
            // tensor_float_destroy(loss_diff); // Owned by add op
            // tensor_float_destroy(out); // Owned by dense op chain

        } // End of dataset loop

        if (nan_detected) {
             printf("Epoch %d stopped early due to NaN.\n", j + 1);
             // Optionally, break outer loop too if NaN means unrecoverable
             // break;
        }

        // Optional: Print epoch loss
         if ((j + 1) % 100 == 0 && epoch_batches > 0 && !nan_detected)
         { // Print every 100 epochs if no NaN and batches were processed
            printf("Epoch [%d/%d], Average Loss: %.6f\n", j + 1, num_epochs, epoch_loss_sum / epoch_batches);
             // Also print current weight and bias for debugging
             if(fc1 && fc1->weights && fc1->weights->val && fc1->weights->val->val &&
                fc1->biases && fc1->biases->val && fc1->biases->val->val) {
                 printf("  W: %.4f, b: %.4f\n", fc1->weights->val->val[0], fc1->biases->val->val[0]);
             }
         } else if ((j + 1) % 100 == 0) {
             printf("Epoch [%d/%d], No valid batches processed or NaN detected.\n", j + 1, num_epochs);
         }


    } // End of epoch loop

    printf("Training completed\n");

    // --- Inference ---
    float cel = 4.0f; // Test value
    float test_data[] = {cel};
    int test_shape[] = {1, 1}; // Match input shape [1, 1]
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

    if (out1 && out1->val && out1->val->val && !isnan(out1->val->val[0]))
    {
        printf("The conversion of %.1f degrees celcius to fahrenheit is %.4f\n", cel, out1->val->val[0]);
        // Expected output for 4 C should be near 39.2 F (4 * 1.8 + 32)
         // Check the learned parameters
         if(fc1 && fc1->weights && fc1->weights->val && fc1->weights->val->val &&
            fc1->biases && fc1->biases->val && fc1->biases->val->val) {
             printf("  Learned W: %.4f (target ~1.8), Learned b: %.4f (target ~32.0)\n",
                    fc1->weights->val->val[0], fc1->biases->val->val[0]);
         }
    } else
    {
        printf("Inference failed or produced invalid/NaN tensor.\n");
        if (out1 && out1->val && out1->val->val && isnan(out1->val->val[0])) {
             printf("  Result was NaN.\n");
        }
    }

    // --- Clean up ---
    // Destroy inference tensors
    // Note: out1 is intermediate, owned by dense_forward ops. We only destroy 'test'.
    // tensor_float_destroy(out1); // Don't destroy - owned by operation
    tensor_float_destroy(test);

    // Destroy model, optimizer, and dataset
    dense_float_destroy(fc1);
    // Call destroy via the function pointer in the base struct
    sgd->base_optimizer.destroy((Optimizer_float*)sgd);
    dataset->base_loader.destroy((DataLoader_float_float*)dataset);

    printf("Cleanup finished.\n");

    return 0;
}