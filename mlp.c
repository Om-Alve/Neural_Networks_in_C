#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define ROWS 4;
#define COLS 2;

float** ones(int row,int col){
	float** mat = (float **)malloc(row*sizeof(float*));
	for(int i=0;i<row;i++){
		mat[i] = (float *)malloc(col* sizeof(float));
	}
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			mat[i][j] = 0.1;
		}
	}
	return mat;
}

float** dot(float** a,float** b,int r1,int c1,int r2,int c2){
	int m = r1,n = c2;
	float** ans = (float**)malloc(m * sizeof(float*));
	for(int i=0;i<m;i++){
		ans[i] = (float *)malloc(n * sizeof(float));
	}
	for(int i = 0;i<r1;i++){
		for(int j = 0;j<c2;j++){
			ans[i][j] = 0;
			for(int k=0;k<r2;k++){
				ans[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return ans;
}


float** scalar_mul(float** x,float y, int r, int c) {
    float** ans = (float**)malloc(r * sizeof(float*));
    for(int i = 0; i < r; i++) {
        ans[i] = (float*)malloc(c * sizeof(float));
    }
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            ans[i][j] = x[i][j]*y;
        }
    }
    return ans;
}


float** sum(float** a,float** b,int r,int c){
	float** ans = (float**)malloc(r * sizeof(float*));
	for(int i=0;i<r;i++){
		ans[i] = (float *)malloc(c * sizeof(float));
	}
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			ans[i][j] = a[i][j] + b[i][j];
		}
	}
	return ans;
}

float** sub(float** a,float** b,int r,int c){
	float** ans = (float**)malloc(r * sizeof(float*));
	for(int i=0;i<r;i++){
		ans[i] = (float *)malloc(c * sizeof(float));
	}
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			ans[i][j] = a[i][j] - b[i][j];
		}
	}
	return ans;
}

float** tanh_activation(float** x,int r,int c){
	float** ans = (float**)malloc(r * sizeof(float*));
	for(int i=0;i<r;i++){
		ans[i] = (float *)malloc(c * sizeof(float));
	}
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			ans[i][j] = tanh(x[i][j]);
		}
	}
	return ans;
}


float** mul(float** a,float** b,int r,int c){
	float** ans = (float**)malloc(r * sizeof(float*));
	for(int i=0;i<r;i++){
		ans[i] = (float *)malloc(c * sizeof(float));
	}
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			ans[i][j] = a[i][j] * b[i][j];
		}
	}
	return ans;
}

float mean_squared_loss(float** y_true,float** y_pred,int r,int c){
	float **ans = sub(y_true,y_pred,r,c);
	float loss = 0.0;
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			loss += ans[i][j] * ans[i][j];
		}
	}
	return loss/(r*c);
}

float** derivative_tanh(float** x, int r, int c) {
    float** ans = (float**)malloc(r * sizeof(float*));
    for(int i = 0; i < r; i++) {
        ans[i] = (float*)malloc(c * sizeof(float));
    }
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            ans[i][j] = 1 - (x[i][j] * x[i][j]); // derivative of tanh(x) = 1 - tanh^2(x)
        }
    }
    return ans;
}

float** transpose(float** x,int r,int c){
	float** ans = (float**)malloc(c * sizeof(float*));
    for(int i = 0; i < c; i++) {
        ans[i] = (float*)malloc(r * sizeof(float));
    }
	for(int i=0;i<c;i++){
		for(int j=0;j<r;j++){
			ans[i][j] = x[j][i];
		}
	}
	return ans;
}

void display(float** a,int r,int c){
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			printf("%f ",a[i][j]);
		}
		printf("\n");
	}
}



int main(){
	float** X = (float**)malloc(4 * sizeof(float*));
	for(int i=0; i<4; i++){
		X[i] = (float*)malloc(2 * sizeof(float)); 
	}

	// Initialize values  
	X[0][0] = 0.0f;
	X[0][1] = 0.0f;

	X[1][0] = 0.0f;
	X[1][1] = 1.0f;

	X[2][0] = 1.0f;
	X[2][1] = 0.0f;  

	X[3][0] = 1.0f;
	X[3][1] = 1.0f;
	float** y = (float**)malloc(4 * sizeof(float*));
	for(int i=0; i<4; i++){
		y[i] = (float*)malloc(1 * sizeof(float)); 
	}
	y[0][0] = 0;
	y[1][0] = 1;
	y[2][0] = 1;
	y[3][0] = 0;
	float** W1 = ones(2,3);
	float** b1 = ones(4,3);
	float** W2 = ones(3,1);
	float** b2 = ones(4,1);
	for(int i=0;i<21;i++){
		float** h = sum(dot(X,W1,4,2,2,3),b1,4,3);
		h = tanh_activation(h,4,3);
		float** out = sum(dot(h,W2,4,3,3,1),b2,4,1);
		float loss =mean_squared_loss(y,out,4,1);
		// display(out,4,1);
		printf("Epoch : %d Loss : %f\n",i,loss);
		float** dL_dout = scalar_mul(sub(out,y,4,1),2,4,1);
		float** dout_dW2 = h;
		float** dL_dW2 = dot(transpose(dout_dW2,4,3),dL_dout,3,4,4,1);
		float** dL_db2 = dL_dout;

		float** dout_dh = W2;
		float** dL_dh = dot(dL_dout,transpose(dout_dh,3,1),4,1,1,3);
		float** dL_dz = mul(dL_dh,derivative_tanh(h,4,3),4,3);
		
		float** dz_dW1 = X;
		float** dL_dW1 = dot(transpose(dz_dW1,4,2),dL_dz,2,4,4,3);
		float** dL_db1 = dL_dz;

		float lr = 0.1;

		W1 = sub(W1,scalar_mul(dL_dW1,lr,2,3),2,3);
		b1 = sub(b1,scalar_mul(dL_db1,lr,4,3),4,3);

		W2 = sub(W2,scalar_mul(dL_dW2,lr,3,1),3,1);
		b2 = sub(b2,scalar_mul(dL_db2,lr,4,1),4,1);
	}
	float** h = sum(dot(X,W1,4,2,2,3),b1,4,3);
	h = tanh_activation(h,4,3);
	float** out = sum(dot(h,W2,4,3,3,1),b2,4,1);
	printf("Expected output: \n");
	display(y,4,1);
	printf("Model's output: \n");
	display(out,4,1);
	return 0;
}
