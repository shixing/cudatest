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
matrixMulCUDA(float *C, float *A, float *B,
    const int M, const int N, const int K) {
  int cx = threadIdx.x + blockIdx.x*blockDim.x;
  int cy = threadIdx.y + blockIdx.y*blockDim.y;
  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0.f;
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
    for (int ak = threadIdx.y, bk = threadIdx.x; ;
        ak += BLOCK_SIZE, bk += BLOCK_SIZE) {
      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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

      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub += As[k][threadIdx.x] * Bs[threadIdx.y][k];
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
    C[cy*M+cx] = Csub;
  }
}




int main() {
  FILE *fp = NULL;
  void *db_buf, *pad_buf, *dist_buf;
void *d_db_buf, *d_pad_buf, *d_dist_buf, *d_dist_buf_1, *d_dist_buf_2;
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
    checkCudaError(cudaMalloc(&d_dist_buf, count*sizeof(float)));
    checkCudaError(cudaMalloc(&d_dist_buf_1, count*sizeof(float)));
    checkCudaError(cudaMalloc(&d_dist_buf_2, count*sizeof(float)));
    checkCudaError(cudaMemcpy(d_dist_buf, dist_buf, count*sizeof(float),
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
          (float*)d_dist_buf_1, (float*)d_db_buf, (float*)d_pad_buf,
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

cublasSgeam(cublasHandle,
	    CUBLAS_OP_N, CUBLAS_OP_N,
	      voc, batch,
	      &alpha, (float*)d_dist_buf_1, voc,
	      &beta, (float*)d_dist_buf_2,voc, 
	      (float*)d_dist_buf,voc);
    checkCudaError(cudaGetLastError());

    cublasSasum(cublasHandle, 1, (float*)d_dist_buf_1, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout << result << '\n' ;
    cublasSasum(cublasHandle, 1, (float*)d_dist_buf_2, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout << result << '\n' ;
    cublasSasum(cublasHandle, voc * batch, (float*)d_dist_buf, 1, &result);
    checkCudaError(cudaGetLastError());
    std::cout << result << '\n' ;


  }


  return 0;
}
