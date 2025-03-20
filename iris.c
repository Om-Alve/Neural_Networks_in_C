#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  // Loading the Iris dataset

  double **features = (double **)malloc(150 * sizeof(double *));
  double **labels = (double **)malloc(150 * sizeof(double *));
  for (int i = 0; i < 150; i++) {
    features[i] = (double *)malloc(4 * sizeof(double));
    labels[i] = (double *)malloc(sizeof(double));
  }
  FILE *file = fopen("Iris.csv", "r");
  char buffer[100];
  fgets(buffer, sizeof(buffer), file);
  int idx = 0;
  while (fgets(buffer, sizeof(buffer), file)) {
    char *token = strtok(buffer, ",");
    token = strtok(NULL, ",");
    for (int i = 0; i < 4; i++) {
      features[idx][i] = atof(token);
      token = strtok(NULL, ",");
    }
    if (strcmp(token, "Iris-setosa\n") == 0) {
      labels[idx][0] = 0;
    } else if (strcmp(token, "Iris-versicolor\n") == 0) {
      labels[idx][0] = 1;
    } else {
      labels[idx][0] = 2;
    }
    idx++;
  }
  fclose(file);
  printf("Dataset Loaded!\n");

  // Converting arrays to Tensors

  Tensor *X = to_tensor(features, 150, 4);
  Tensor *y = to_tensor(labels, 150, 1);

  // Free original arrays after copying their data into tensors
  for (int i = 0; i < 150; i++) {
    free(features[i]);
    free(labels[i]);
  }
  free(features);
  free(labels);

  // Initializing parameters

  double lr = 0.1;
  Tensor *W1 = kaiming_init(4, 100, sqrt(2));
  Tensor *b1 = zeros(150, 100);
  Tensor *W2 = kaiming_init(100, 100, sqrt(2));
  Tensor *b2 = zeros(150, 100);
  Tensor *W3 = kaiming_init(100, 3, sqrt(2));
  Tensor *b3 = zeros(150, 3);

  // Training

  Tensor *h1, *h1_act, *h2, *h2_act, *out, *probs, *preds;
  Tensor *dL_dprobs, *dL_dW3, *dL_db3, *dout_dh2, *dL_dh2, *dL_dz2;
  Tensor *dL_dW2, *dL_db2, *dL_dh1, *dL_dz1, *dL_dW1, *dL_db1;
  Tensor *tmp; // for temporary tensors in parameter updates

  for (int i = 0; i < 401; i++) {
    // Forward Pass

    Tensor *dot1 = dot(X, W1);
    h1 = sum(dot1, b1);
    free_tensor(dot1);

    h1_act = Relu_activation(h1);
    free_tensor(h1);

    Tensor *dot2 = dot(h1_act, W2);
    h2 = sum(dot2, b2);
    free_tensor(dot2);

    h2_act = Relu_activation(h2);
    free_tensor(h2);

    Tensor *dot3 = dot(h2_act, W3);
    out = sum(dot3, b3);
    free_tensor(dot3);

    probs = softmax_activation(y, out);
    free_tensor(out);

    // Calculating and printing the loss

    double loss = cross_entropy_loss(y, probs);
    if (i % 100 == 0) {
      printf("Epoch : %d Loss : %f\n", i, loss);
    }

    // Backpropagation
    dL_dprobs = Softmax_crossentropy_backprop(probs, y);
    free_tensor(probs);

    // For W3 update:
    // dL_dW3 = dot(transpose(h2_act), dL_dprobs)
    Tensor *h2_act_T = transpose(h2_act);
    dL_dW3 = dot(h2_act_T, dL_dprobs);
    free_tensor(h2_act_T);
    // dL_db3 is just dL_dprobs (assuming proper broadcasting in your tensor
    // lib)
    dL_db3 = to_tensor(dL_dprobs->data, dL_dprobs->r,
                       dL_dprobs->c); // create a copy if needed

    // dout/dh2 = W3, so
    Tensor *W3_T = transpose(W3); // using transpose(W3) to match dimensions
    dL_dh2 = dot(dL_dprobs, W3_T);
    free_tensor(W3_T);

    // dL_dz2 = dL_dh2 * derivative_relu(h2_act)
    Tensor *deriv_h2 = derivative_relu(h2_act);
    dL_dz2 = mul(dL_dh2, deriv_h2);
    free_tensor(deriv_h2);
    free_tensor(dL_dh2);

    // dL_dW2 = dot(transpose(h1_act), dL_dz2)
    Tensor *h1_act_T = transpose(h1_act);
    dL_dW2 = dot(h1_act_T, dL_dz2);
    free_tensor(h1_act_T);
    // dL_db2 = dL_dz2
    dL_db2 = to_tensor(dL_dz2->data, dL_dz2->r,
                       dL_dz2->c); // create a copy if needed

    // dL_dh1 = dot(dL_dz2, transpose(W2))
    Tensor *W2_T = transpose(W2);
    dL_dh1 = dot(dL_dz2, W2_T);
    free_tensor(W2_T);
    free_tensor(dL_dz2);

    // dL_dz1 = dL_dh1 * derivative_relu(h1_act)
    Tensor *deriv_h1 = derivative_relu(h1_act);
    dL_dz1 = mul(dL_dh1, deriv_h1);
    free_tensor(deriv_h1);
    free_tensor(dL_dh1);

    // dL_dW1 = dot(transpose(X), dL_dz1)
    Tensor *X_T = transpose(X);
    dL_dW1 = dot(X_T, dL_dz1);
    free_tensor(X_T);
    // dL_db1 = dL_dz1
    dL_db1 = to_tensor(dL_dz1->data, dL_dz1->r,
                       dL_dz1->c); // create a copy if needed

    free_tensor(dL_dz1);
    // Now update parameters with gradient descent
    // For each parameter, multiply the gradient by lr, subtract from the
    // parameter, free the old parameter, and free temporary tensors

    // Update W1
    tmp = scalar_mul(dL_dW1, lr);
    Tensor *new_W1 = sub(W1, tmp);
    free_tensor(tmp);
    free_tensor(W1);
    W1 = new_W1;
    free_tensor(dL_dW1);

    // Update b1
    tmp = scalar_mul(dL_db1, lr);
    Tensor *new_b1 = sub(b1, tmp);
    free_tensor(tmp);
    free_tensor(b1);
    b1 = new_b1;
    free_tensor(dL_db1);

    // Update W2
    tmp = scalar_mul(dL_dW2, lr);
    Tensor *new_W2 = sub(W2, tmp);
    free_tensor(tmp);
    free_tensor(W2);
    W2 = new_W2;
    free_tensor(dL_dW2);

    // Update b2
    tmp = scalar_mul(dL_db2, lr);
    Tensor *new_b2 = sub(b2, tmp);
    free_tensor(tmp);
    free_tensor(b2);
    b2 = new_b2;
    free_tensor(dL_db2);

    // Update W3
    tmp = scalar_mul(dL_dW3, lr);
    Tensor *new_W3 = sub(W3, tmp);
    free_tensor(tmp);
    free_tensor(W3);
    W3 = new_W3;
    free_tensor(dL_dW3);

    // Update b3
    tmp = scalar_mul(dL_db3, lr);
    Tensor *new_b3 = sub(b3, tmp);
    free_tensor(tmp);
    free_tensor(b3);
    b3 = new_b3;
    free_tensor(dL_db3);

    // Free intermediate tensors from forward/backward pass that are still
    // allocated
    free_tensor(h1_act);
    free_tensor(h2_act);
    free_tensor(dL_dprobs);
  }

  // Final prediction
  // Recompute the forward pass using the final parameters
  Tensor *temp_h1 = sum(dot(X, W1), b1);
  Tensor *temp_h1_act = Relu_activation(temp_h1);
  free_tensor(temp_h1);

  Tensor *temp_h2 = sum(dot(temp_h1_act, W2), b2);
  Tensor *temp_h2_act = Relu_activation(temp_h2);
  free_tensor(temp_h2);

  // Use temp_h2_act for prediction
  Tensor *temp_dot = dot(temp_h2_act, W3);
  Tensor *temp_out = sum(temp_dot, b3);
  free_tensor(temp_dot);
  Tensor *temp_probs = softmax_activation(y, temp_out);
  free_tensor(temp_out);
  preds = argmax(temp_probs);
  free_tensor(temp_probs);

  // Clean up temporary tensors from forward pass
  free_tensor(temp_h1_act);
  free_tensor(temp_h2_act);

  printf("Expected output: \n");
  for (int i = 0, j = 0; j < 5; i += 30, j++) {
    printf("%.1f ", y->data[i][0]);
  }
  printf("\n");
  printf("Model's output: \n");
  for (int i = 0, j = 0; j < 5; i += 30, j++) {
    printf("%.1f ", preds->data[i][0]);
  }
  printf("\n");
  printf("Final Accuracy : %f\n", accuracy(y, preds));

  // Free remaining tensors
  free_tensor(X);
  free_tensor(y);
  free_tensor(W1);
  free_tensor(b1);
  free_tensor(W2);
  free_tensor(b2);
  free_tensor(W3);
  free_tensor(b3);
  free_tensor(preds);

  return 0;
}
