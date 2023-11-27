#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct
{
    double **data;
    int r, c;
} Tensor;

Tensor* ones(int r, int c);
Tensor* zeros(int r, int c); 

Tensor* uniform(int r, int c, float lb, float ub);
Tensor* randn(int r, int c);

Tensor* kaiming_init(int fin, int fout, double gain);

Tensor* sum(Tensor* a, Tensor* b);
Tensor* mul(Tensor* a, Tensor* b);
Tensor* sub(Tensor* a, Tensor* b);

Tensor* dot(Tensor* a, Tensor* b);
Tensor* scalar_mul(Tensor* x, double y);

Tensor* derivative_tanh(Tensor* x);  
Tensor* transpose(Tensor* x);

Tensor* argmax(Tensor* x);

Tensor* Relu_activation(Tensor* x);
Tensor* derivative_relu(Tensor* x); 
Tensor* tanh_activation(Tensor* x);

double cross_entropy_loss(Tensor* y_true, Tensor* y_pred); 

Tensor* softmax_activation(Tensor* y_true, Tensor* y_pred);

Tensor* Softmax_crossentropy_backprop(Tensor* y_pred, Tensor* y_true); 

double mean_squared_loss(Tensor* y_true, Tensor* y_pred);

Tensor* to_tensor(double** array, int rows, int cols);
void free_tensor(Tensor* tensor);  

void display(Tensor* m);

Tensor *ones(int r, int c)
{
    Tensor *m = (Tensor *)malloc(sizeof(Tensor));
    m->r = r;
    m->c = c;
    double **mat = (double **)malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++)
    {
        mat[i] = (double *)malloc(c * sizeof(double));
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            mat[i][j] = 1;
        }
    }
    m->data = mat;
    return m;
}

Tensor *zeros(int r, int c)
{
    Tensor *m = (Tensor *)malloc(sizeof(Tensor));
    m->r = r;
    m->c = c;
    double **mat = (double **)malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++)
    {
        mat[i] = (double *)malloc(c * sizeof(double));
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            mat[i][j] = 0;
        }
    }
    m->data = mat;
    return m;
}

Tensor* uniform(int r, int c, float lb, float ub) {

  Tensor* m = (Tensor*) malloc(sizeof(Tensor));
  m->r = r;
  m->c = c;
  
  double** mat = (double**) malloc(r * sizeof(double*));
  for(int i = 0; i < r; i++) {
    mat[i] = (double*) malloc(c * sizeof(double)); 
  }

  for(int i = 0; i < r; i++) {
    for(int j = 0; j < c; j++) {
      double range = ub - lb;
      mat[i][j] = lb + (rand() / (double)RAND_MAX) * range;
    }
  }

  m->data = mat;
  
  return m; 
}

Tensor *randn(int r, int c)
{
    Tensor *m = (Tensor *)malloc(sizeof(Tensor));
    m->r = r;
    m->c = c;
    double **mat = (double **)malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++)
    {
        mat[i] = (double *)malloc(c * sizeof(double));
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            mat[i][j] = (double)rand() / (double)RAND_MAX ;
        }
    }
    m->data = mat;
    return m;
}

Tensor* kaiming_init(int fin,int fout,double gain){
    return scalar_mul(randn(fin, fout), gain * sqrt(3/fin));
}

Tensor *sum(Tensor *a, Tensor *b)
{
    if (a->r != b->r || a->c != b->c)
    {
        printf("Tensor dimensions are different. Cannot perform addition.\n");
        return NULL;
    }

    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = a->r;
    ans->c = a->c;
    ans->data = (double **)malloc(ans->r * sizeof(double *));

    for (int i = 0; i < ans->r; i++)
    {
        ans->data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            ans->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return ans;
}

Tensor *mul(Tensor *a, Tensor *b)
{
    if (a->r != b->r || a->c != b->c)
    {
        printf("Tensor dimensions are different. Cannot perform multiplication.\n");
        return NULL;
    }

    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = a->r;
    ans->c = a->c;
    ans->data = (double **)malloc(ans->r * sizeof(double *));

    for (int i = 0; i < ans->r; i++)
    {
        ans->data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            ans->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    return ans;
}

Tensor *sub(Tensor *a, Tensor *b)
{
    if (a->r != b->r || a->c != b->c)
    {
        printf("Tensor dimensions are different. Cannot perform subtraction.\n");
        return NULL;
    }

    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = a->r;
    ans->c = a->c;
    ans->data = (double **)malloc(ans->r * sizeof(double *));

    for (int i = 0; i < ans->r; i++)
    {
        ans->data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            ans->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return ans;
}

Tensor *dot(Tensor *a, Tensor *b)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = a->r, ans->c = b->c;
    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
    }
    for (int i = 0; i < ans->r; i++)
    {
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = 0;
            for (int k = 0; k < a->c; k++)
            {
                data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    ans->data = data;
    return ans;
}

Tensor *scalar_mul(Tensor *x, double y)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = x->c;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = x->data[i][j] * y;
        }
    }
    ans->data = data;
    return ans;
}

Tensor *derivative_tanh(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = x->c;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = 1 - (x->data[i][j] * x->data[i][j]); // derivative of tanh(x) = 1 - tanh^2(x)
        }
    }
    ans->data = data;
    return ans;
}

Tensor *transpose(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->c;
    ans->c = x->r;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = x->data[j][i];
        }
    }
    ans->data = data;
    return ans;
}

Tensor *argmax(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = 1;
    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        int maxidx = 0;
        for (int j = 1; j < x->c; j++)
        {
            if (x->data[i][maxidx] < x->data[i][j])
            {
                maxidx = j;
            }
        }
        data[i][0] = (double)maxidx;
    }
    ans->data = data;
    return ans;
}

double accuracy(Tensor* y_true,Tensor* y_pred){
    double acc = 0.0;
    for (int i = 0; i < y_true->r; i++)
    {
        if(y_true->data[i][0] == y_pred->data[i][0]){
            acc++;
        }
    }
    return acc/y_true->r * 100.0;
}

Tensor *Relu_activation(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = x->c;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            if (x->data[i][j] <= 0)
            {
                data[i][j] = 0;
            }
            else
            {
                data[i][j] = x->data[i][j];
            }
        }
    }
    ans->data = data;
    return ans;
}

Tensor *derivative_relu(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = x->c;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            if (x->data[i][j] <= 0)
            {
                data[i][j] = 0;
            }
            else
            {
                data[i][j] = 1;
            }
        }
    }
    ans->data = data;
    return ans;
}

Tensor *tanh_activation(Tensor *x)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = x->r;
    ans->c = x->c;

    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = tanh(x->data[i][j]);
        }
    }
    ans->data = data;
    return ans;
}

double cross_entropy_loss(Tensor *y_true, Tensor *y_pred)
{
    double loss = 0;
    for (int i = 0; i < y_true->r; i++)
    {
        for (int j = 0; j < y_true->c; j++)
        {
            loss += log(y_pred->data[i][(int)y_true->data[i][0]] + 0.000000001);
        }
    }
    return -loss / y_true->r;
}

Tensor *softmax_activation(Tensor *y_true, Tensor *y_pred)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = y_pred->r;
    ans->c = y_pred->c;
    double maxVal;
    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        maxVal = data[i][0];
        for (int j = 1; j < ans->c; j++)
        {
            if (data[i][j] > maxVal)
                maxVal = data[i][j];
        }
        double exp_sum = 0.0;
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = exp(y_pred->data[i][j] - maxVal);
            exp_sum += data[i][j];
        }
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] /= exp_sum;
        }
    }
    ans->data = data;
    return ans;
}

Tensor *Softmax_crossentropy_backprop(Tensor *y_pred, Tensor *y_true)
{
    Tensor *ans = (Tensor *)malloc(sizeof(Tensor));
    ans->r = y_pred->r;
    ans->c = y_pred->c;
    double **data = (double **)malloc(ans->r * sizeof(double *));
    for (int i = 0; i < ans->r; i++)
    {
        data[i] = (double *)malloc(ans->c * sizeof(double));
        for (int j = 0; j < ans->c; j++)
        {
            data[i][j] = y_pred->data[i][j];
            if (j == y_true->data[i][0])
            {
                data[i][j] -= 1;
            }
        }
    }
    ans->data = data;
    return ans;
}

double mean_squared_loss(Tensor *y_true, Tensor *y_pred)
{
    if (y_true->r != y_pred->r || y_true->c != y_pred->c)
    {
        printf("Tensor dimensions are different. Cannot calculate loss.\n");
        return -1.0;
    }

    double loss = 0.0;
    for (int i = 0; i < y_true->r; i++)
    {
        for (int j = 0; j < y_true->c; j++)
        {
            double diff = y_true->data[i][j] - y_pred->data[i][j];
            loss += diff * diff;
        }
    }
    return loss / (y_true->r * y_true->c);
}

Tensor *to_tensor(double **array, int rows, int cols)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->r = rows;
    tensor->c = cols;
    tensor->data = (double **)malloc(rows * sizeof(double *));

    for (int i = 0; i < rows; i++)
    {
        tensor->data[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++)
        {
            tensor->data[i][j] = array[i][j];
        }
    }

    return tensor;
}
void free_tensor(Tensor *tensor)
{
    for (int i = 0; i < tensor->r; i++)
    {
        free(tensor->data[i]);
    }
    free(tensor->data);
    free(tensor);
}

void display(Tensor *m)
{
    for (int i = 0; i < m->r; i++)
    {
        for (int j = 0; j < m->c; j++)
        {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}