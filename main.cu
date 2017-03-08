#include "util.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
//#include <glog/logging.h>
#include <cuda_runtime.h>

const char* db_input = "./data/d_Db_input.txt.bin";
const char* pad_input = "./data/d_ht_pad_input.txt.bin";
const char* dist_input = "./data/d_outputdist_input.txt.bin";
const char* dist_output = "./data/d_outputdist_output.txt.bin";

const int batch = 12;
const int voc = 50000;
const int embed = 1001;

const int block_size = 32;

#define LOG(INFO) std::cout
#define CHECK assert


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A^T * B
 * A = k x m
 * B = k x n
 * All matrice are column major
 */
template <int BLOCK_SIZE>
__global__ void
matrixMulCUDA(float *C, float *C_input, float *A, float *B,
    const int M, const int N, const int K) {
  int cx = threadIdx.x + blockIdx.x*blockDim.x;
  int cy = threadIdx.y + blockIdx.y*blockDim.y;
  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0.f;
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int ak = threadIdx.y, bk = threadIdx.x; ;
       ak += BLOCK_SIZE, bk += BLOCK_SIZE) {
    
    /*// Load the matrices from device memory*/
    /*// to shared memory; each thread loads*/
    /*// one element of each matrix*/
    if (ak < K && cx < M) {
      As[threadIdx.y][threadIdx.x] = A[cx*K+ak]; //As = A
    } else {
      As[threadIdx.y][threadIdx.x] = 0; 
    }
    if (bk < K && cy < N){
      Bs[threadIdx.y][threadIdx.x] = B[cy*K+bk]; //Bs = B^T
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0; 
    }
    
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    
    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    
    if (cx < M && cy < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
	Csub += As[k][threadIdx.x] * Bs[threadIdx.y][k];
      } 
    }
    
    /*// Synchronize to make sure that the preceding*/
    /*// computation is done before loading two new*/
    /*// sub-matrices of A and B in the next iteration*/
    __syncthreads();
    
    if (ak / BLOCK_SIZE == K/BLOCK_SIZE){
      break;
    }
    
  }
  
  // Write the block sub-matrix to device memory;
  // each thread writes one element
  
  if (cx < M && cy < N) {
    if (C_input[cy*M+cx] > 0){
      C[cy*M+cx] = Csub;
    } else {
      C[cy*M+cx] = -1000;
    }
    
  }
}


// C = A - B if A[i] == -1000, C[i] == 0;
// <<<1, 1024>>>
__global__ 
void special_matrix_sum(float *C, float *A, float *B, int count) {
  for (int i = threadIdx.x; i< count; i += blockDim.x) {
    if (A[i] == -1000){
      C[i] = 0;
    } else {
      C[i] = A[i] - B[i];
    }
  }
}
  




template<typename dType>
void print_matrix_gpu(dType *d_matrix,int rows,int cols, int row_start, int row_end, int col_start, int col_end) {
    dType * h_matrix = (dType *)malloc(rows*cols*sizeof(dType));
    cudaMemcpy(h_matrix, d_matrix, rows*cols*sizeof(dType), cudaMemcpyDeviceToHost);
    for(int i=row_start; i<row_end; i++) {
        for(int j=col_start; j<col_end; j++) {
	  std::cout << h_matrix[i + j*rows] << " ";
        }
	std::cout << "\n";
    }
    std::cout << "\n";
    free(h_matrix);
}



int main() {
  FILE *fp = NULL;
  void *db_buf, *pad_buf, *dist_buf;
  void *d_db_buf, *d_pad_buf, *d_dist_buf_input,*d_dist_buf_res,*d_dist_buf_output, *d_dist_buf_1, *d_dist_buf_2;
  {
CHECK(fp = fopen(db_input, "rb"));
    int db_input_meta[2];
    CHECK(fread(db_input_meta, sizeof(int), 2, fp) == 2);
    CHECK(db_input_meta[0] == embed); //<< db_input_meta[0];
    CHECK(db_input_meta[1] == voc);// << db_input_meta[1];
    int count = embed*voc;
    db_buf = malloc(count*sizeof(float));
    CHECK(fread(db_buf, sizeof(float), count, fp) == count);
    checkCudaError(cudaMalloc(&d_db_buf, count*sizeof(float)));
    checkCudaError(cudaMemcpy(d_db_buf, db_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  {
CHECK(fp = fopen(pad_input, "rb"));
    int pad_input_meta[2];
    CHECK(fread(pad_input_meta, sizeof(int), 2, fp) == 2);
CHECK(pad_input_meta[0] == embed) ;//<< pad_input_meta[0];
CHECK(pad_input_meta[1] == batch);// << pad_input_meta[1];
    int count = embed*batch;
    pad_buf = malloc(count*sizeof(float));
    CHECK(fread(pad_buf, sizeof(float), count, fp) == count);
    checkCudaError(cudaMalloc(&d_pad_buf, count*sizeof(float)));
    checkCudaError(cudaMemcpy(d_pad_buf, pad_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  {
CHECK(fp = fopen(dist_input, "rb"));
    int dist_input_meta[2];
    CHECK(fread(dist_input_meta, sizeof(int), 2, fp) == 2);
CHECK(dist_input_meta[0] == voc);// << dist_input_meta[0];
CHECK(dist_input_meta[1] == batch);// << dist_input_meta[1];
    int count = voc*batch;
    dist_buf = malloc(count*sizeof(float));
    CHECK(fread(dist_buf, sizeof(float), count, fp) == count);
    checkCudaError(cudaMalloc(&d_dist_buf_input, count*sizeof(float)));
    checkCudaError(cudaMalloc(&d_dist_buf_1, count*sizeof(float)));
    checkCudaError(cudaMalloc(&d_dist_buf_2, count*sizeof(float)));
    checkCudaError(cudaMalloc(&d_dist_buf_res, count*sizeof(float)));
    checkCudaError(cudaMemcpy(d_dist_buf_input, dist_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_dist_buf_1, dist_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_dist_buf_2, dist_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));

  }

  {
CHECK(fp = fopen(dist_output, "rb"));
    int dist_input_meta[2];
    CHECK(fread(dist_input_meta, sizeof(int), 2, fp) == 2);
CHECK(dist_input_meta[0] == voc);// << dist_input_meta[0];
CHECK(dist_input_meta[1] == batch);// << dist_input_meta[1];
    int count = voc*batch;
    dist_buf = malloc(count*sizeof(float));
    CHECK(fread(dist_buf, sizeof(float), count, fp) == count);
    checkCudaError(cudaMalloc(&d_dist_buf_output, count*sizeof(float)));
    checkCudaError(cudaMemcpy(d_dist_buf_output, dist_buf, count*sizeof(float),
                              cudaMemcpyHostToDevice));
  }


  
  {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    dim3 threads(block_size, block_size);
    dim3 grid((voc+threads.x-1)/threads.x,
              (batch+threads.y-1)/threads.y);
    int Test = 100;
    checkCudaError(cudaEventRecord(start, NULL));
    for (int i = 0; i < Test; i++) {
      matrixMulCUDA<block_size><<<grid, threads>>>(
          (float*)d_dist_buf_1, (float*)d_dist_buf_input, (float*)d_db_buf, (float*)d_pad_buf,
          voc, batch, embed);
    }
    checkCudaError(cudaEventRecord(stop, NULL));
    checkCudaError(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));
    msecTotal /= Test;
    double flopsPerMatrixMul = 2.0 * voc * batch * embed;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f);
    LOG(INFO) << "Performance= " << gigaFlops << " GFlop/s, "
              << "Time= " << msecTotal << " msec, ";
    checkCudaError(cudaGetLastError());
  }

  {
    cublasHandle_t cublasHandle;
    checkCublasError(cublasCreate(&cublasHandle));
    float alpha = 1.f, beta = 0.f;
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    dim3 threads(block_size, block_size);
    dim3 grid((voc+threads.x-1)/threads.x,
              (batch+threads.y-1)/threads.y);
    int Test = 100;
    checkCudaError(cudaEventRecord(start, NULL));
    for (int i = 0; i < Test; i++) {
      cublasSgemm(cublasHandle,
                  CUBLAS_OP_T, CUBLAS_OP_N,
                  voc, batch, embed, &alpha,
                  (float*)d_db_buf, embed,
                  (float*)d_pad_buf, embed,
                  &beta,
                  (float*)d_dist_buf_2, voc);
    }
    checkCudaError(cudaEventRecord(stop, NULL));
    checkCudaError(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));
    msecTotal /= Test;
    double flopsPerMatrixMul = 2.0 * voc * batch * embed;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f);
    LOG(INFO) << "Performance= " << gigaFlops << " GFlop/s, "
              << "Time= " << msecTotal << " msec, ";
    checkCudaError(cudaGetLastError());
  }

  {
    //check d_dist_buf_1 == d_dist_buf_2
    cublasHandle_t cublasHandle;
    checkCublasError(cublasCreate(&cublasHandle));
float alpha = 1.f, beta = -1.f, result = 1.0f;


 special_matrix_sum<<<1,1024>>>((float*)d_dist_buf_res,(float*) d_dist_buf_1,(float*) d_dist_buf_2,voc*batch);

    cublasSasum(cublasHandle, 1, (float*)d_dist_buf_1, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout <<"\nFirst element by MatrixMul: "<< result << '\n' ;

    cublasSasum(cublasHandle, 1, (float*)d_dist_buf_2, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout <<"First element by cublas: " << result << '\n' ;

    cublasSasum(cublasHandle, 1, (float*)d_dist_buf_output, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout <<"\nFirst element by d_dist_buf_output: "<< result << '\n' ;

    cublasSasum(cublasHandle, voc * batch, (float*)d_dist_buf_res, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout <<"MatrixMul - cublas (sparse): "<< result << '\n' ;

    std::cout << "d_dist_buf_input\n";
    print_matrix_gpu((float*)d_dist_buf_input, voc, batch, 0, 10, 0, batch);

    std::cout << "d_dist_buf_1\n";
    print_matrix_gpu((float*)d_dist_buf_1, voc, batch, 0, 10, 0, batch);

    std::cout << "d_dist_buf_2\n";
    print_matrix_gpu((float*)d_dist_buf_2, voc, batch, 0, 10, 0, batch);

    //    std::cout << "d_dist_buf_output\n";
    //print_matrix_gpu((float*)d_dist_buf_output, voc, batch, 0, 10, 0, batch);


  }


  return 0;
}
