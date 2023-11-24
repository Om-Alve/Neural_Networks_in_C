#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "tensor.h"

int main(){
    double** X_data = (double**)malloc(4 * sizeof(double*));
	for(int i=0; i<4; i++){
		X_data[i] = (double*)malloc(2 * sizeof(double)); 
	}

	// Initialize values  
	X_data[0][0] = 0.0f;
	X_data[0][1] = 0.0f;

	X_data[1][0] = 0.0f;
	X_data[1][1] = 1.0f;

	X_data[2][0] = 1.0f;
	X_data[2][1] = 0.0f;  

	X_data[3][0] = 1.0f;
	X_data[3][1] = 1.0f;
	double** y_data = (double**)malloc(4 * sizeof(double*));
	for(int i=0; i<4; i++){
		y_data[i] = (double*)malloc(1 * sizeof(double)); 
	}
	y_data[0][0] = 0;
	y_data[1][0] = 1;
	y_data[2][0] = 1;
	y_data[3][0] = 0;
    Tensor* X = to_tensor(X_data,4,2);
    Tensor* y = to_tensor(y_data,4,1);
    Tensor* W1 = scalar_mul(ones(2,4),0.1);
    Tensor* b1 = scalar_mul(ones(4,4),0.1);
    Tensor* W2 = scalar_mul(ones(4,1),0.1);
    Tensor* b2 = scalar_mul(ones(4,1),0.1);
    Tensor*h,*out;
    for(int i=0;i<21;i++){
        h = sum(dot(X,W1),b1);
        h = tanh_activation(h);
        out = sum(dot(h,W2),b2);
        double loss = mean_squared_loss(y,out);
        printf("Epoch : %d Loss : %f\n",i,loss);
        Tensor* dL_dout = scalar_mul(sub(out,y),2);
        Tensor* dout_dW2 = h;
        Tensor* dL_dW2 = dot(transpose(dout_dW2),dL_dout);
        Tensor* dL_db2 = dL_dout;
        Tensor* dout_dh = W2;
        Tensor* dL_dh = dot(dL_dout,transpose(dout_dh));
        Tensor* dL_dz = mul(dL_dh,derivative_tanh(h));
        Tensor* dz_dW1 = X;
        Tensor* dL_dW1 = dot(transpose(dz_dW1),dL_dz);
        Tensor* dL_db1 = dL_dz;

        double lr = 0.1;

        W1 = sub(W1,scalar_mul(dL_dW1,lr));
        b1 = sub(b1,scalar_mul(dL_db1,lr));
        W2 = sub(W2,scalar_mul(dL_dW2,lr));
        b2 = sub(b2,scalar_mul(dL_db2,lr));
    }
    printf("Expected output: \n");
	display(y);
	printf("Model's output: \n");
	display(out);

    free_tensor(X);
    free_tensor(y);
    free_tensor(W1);
    free_tensor(b1);
    free_tensor(W2);
    free_tensor(b2);
    free_tensor(h);
    free_tensor(out);
    return 0;
}