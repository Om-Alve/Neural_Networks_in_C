#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tensor.h"

int main()
{
    // Loading the Iris dataset

    double **features = (double **)malloc(150 * sizeof(double *));
    double **labels = (double **)malloc(150 * sizeof(double *));
    for (int i = 0; i < 150; i++)
    {
        features[i] = (double *)malloc(4 * sizeof(double));
        labels[i] = (double *)malloc(sizeof(double));
    }
    FILE *file = fopen("iris.csv", "r");
    char buffer[100];
    fgets(buffer, sizeof(buffer), file);
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file))
    {
        char *token = strtok(buffer, ",");
        token = strtok(NULL, ",");
        for (int i = 0; i < 4; i++)
        {
            features[idx][i] = atof(token);
            token = strtok(NULL, ",");
        }
        if (strcmp(token, "Iris-setosa\n") == 0)
        {
            labels[idx][0] = 0;
        }
        else if (strcmp(token, "Iris-versicolor\n") == 0)
        {
            labels[idx][0] = 1;
        }
        else
        {
            labels[idx][0] = 2;
        }
        idx++;
    }
    printf("Dataset Loaded!\n");

    // Initializing parameters

    Tensor *X = to_tensor(features, 150, 4);
    Tensor *y = to_tensor(labels, 150, 1);

    double lr = 0.11;
    Tensor *W1 = scalar_mul(ones(4, 100), 0.01);
    Tensor *b1 = scalar_mul(ones(150, 100), 0);
    Tensor *W2 = scalar_mul(ones(100, 100), 0.01);
    Tensor *b2 = scalar_mul(ones(150, 100), 0);
    Tensor *W3 = scalar_mul(ones(100, 3), 0.01);
    Tensor *b3 = scalar_mul(ones(150, 3), 0);

    // Training

    Tensor *h1, *h1_act, *h2, *h2_act, *out, *probs, *preds,*dL_dprobs, 
            *dprobs_dW3, *dL_dW3, *dL_db3, *dout_dh2, *dL_dh2, *dL_dz2, *dz_dW2,
            *dL_dW2, *dL_db2, *dh2_dh1, *dL_dh1, *dL_dz1, *dz_dW1, *dL_dW1, *dL_db1;
    for (int i = 0; i < 101; i++)
    {
        // Forward Pass

        h1 = sum(dot(X, W1), b1);
        h1_act = Relu_activation(h1);
        h2 = sum(dot(h1_act, W2), b2);
        h2_act = Relu_activation(h2);
        out = sum(dot(h2, W3), b3);
        probs = softmax_activation(y, out);

        // Calculating and printing the loss

        double loss = cross_entropy_loss(y, probs);
        if (i % 10 == 0)
        {
            printf("Epoch : %d Loss : %f\n", i, loss);
        }
        // Backpropagation
        dL_dprobs = Softmax_crossentropy_backprop(probs, y);
        dprobs_dW3 = h2_act;
        dL_dW3 = dot(transpose(dprobs_dW3), dL_dprobs);
        dL_db3 = dL_dprobs;
        dout_dh2 = W3;
        dL_dh2 = dot(dL_dprobs, transpose(dout_dh2));
        dL_dz2 = mul(dL_dh2, derivative_relu(h2_act));
        dz_dW2 = h1;
        dL_dW2 = dot(transpose(dz_dW2), dL_dz2);
        dL_db2 = dL_dz2;
        dh2_dh1 = W2;
        dL_dh1 = dot(dL_dh2, transpose(dh2_dh1));
        dL_dz1 = mul(dL_dh1, derivative_relu(h1_act));
        dz_dW1 = X;
        dL_dW1 = dot(transpose(dz_dW1), dL_dz1);
        dL_db1 = dL_dz1;
        // Updating parameters with Gradient descent

        W1 = sub(W1, scalar_mul(dL_dW1, lr));
        b1 = sub(b1, scalar_mul(dL_db1, lr));
        W2 = sub(W2, scalar_mul(dL_dW2, lr));
        b2 = sub(b2, scalar_mul(dL_db2, lr));
        W3 = sub(W3, scalar_mul(dL_dW3, lr));
        b3 = sub(b3, scalar_mul(dL_db3, lr));
    }
    preds = argmax(probs);
    printf("Expected output: \n");
    for (int i = 0,j=0; j < 5; i+=30,j++)
    {
        printf("%.1f ", y->data[i][0]);
    }
    printf("\n");
    printf("Model's output: \n");
    for (int i = 0,j=0; j < 5; i+=30,j++)
    {
        printf("%.1f ", preds->data[i][0]);
    }
    printf("\n");
    printf("Final Loss : %f", cross_entropy_loss(y, probs));
    // Freeing allocated memory

    free_tensor(X);
    free_tensor(y);
    free_tensor(W1);
    free_tensor(b1);
    free_tensor(W2);
    free_tensor(b2);
    free_tensor(W3);
    free_tensor(b3);
    free_tensor(h1);
    free_tensor(h1_act);
    free_tensor(h2);
    free_tensor(h2_act);
    free_tensor(out);
    free_tensor(probs);
    free_tensor(preds);
    free_tensor(dL_dprobs);
    free_tensor(dprobs_dW3);
    free_tensor(dL_dW3);
    free_tensor(dL_db3);
    free_tensor(dout_dh2);
    free_tensor(dL_dh2);
    free_tensor(dL_dz2);
    free_tensor(dz_dW2);
    free_tensor(dL_dW2);
    free_tensor(dL_db2);
    free_tensor(dh2_dh1);
    free_tensor(dL_dh1);
    free_tensor(dL_dz1);
    free_tensor(dz_dW1);
    free_tensor(dL_dW1);
    free_tensor(dL_db1);

    return 0;
}
