#include<stdio.h>
#define SIZE 1024

__global__ void vectorAdd(int *a,int *b,int*c,int n){
  int i=threadIdx.x;
  if(i<n){
    c[i] = a[i] + b[j];
  }
}


int main(){
  int *a,*b,*c;
  cudaMallocManaged(&a,SIZE * sizeof(int));
  cudaMallocManaged(&b,SIZE * sizeof(int));
  cudaMallocManaged(&c,SIZE * sizeof(int));

  for(int i=0;i<SIZE;i++){
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }
  
  vectorAdd<<<1,SIZE>>>(a,b,c,SIZE);
  
  cudaDeviceSyncronized();

  for(int  i=0;i<10;i++){
    printf("%d ",c[i]);
  }
}
