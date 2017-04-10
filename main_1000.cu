#include "util.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
//#include <glog/logging.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

const int batch = 12;
const int voc = 50000;
const int embed = 1001;
 int W = 1000;


#define LOG(INFO) std::cout
#define CHECK assert
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

unsigned int * d_starts, * d_lengths;

// input
//50000 12
template<typename dType>
dType* read_matrix(const char* fn){

  FILE *fp = NULL;
  CHECK(fp = fopen(fn, "rb"));
  int meta[2];
  CHECK(fread(meta, sizeof(int), 2, fp) == 2);
  int count = meta[0] * meta[1];
  dType* h_data = (dType *)malloc(count*sizeof(dType));
  dType* d_data;
  CHECK(fread(h_data, sizeof(dType), count, fp) == count);
  checkCudaError(cudaMalloc(&d_data, count*sizeof(dType)));
  checkCudaError(cudaMemcpy(d_data, h_data, count*sizeof(dType),
			    cudaMemcpyHostToDevice));
  std::cout << fn << " " << meta[0] << " " << meta[1] << "\n";
  return d_data;
}

__device__
unsigned int hash_func_1_gpu(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__device__
unsigned int hash_func_2_gpu(unsigned int key){
    unsigned int c2=0x27d4eb2d; // a prime or an odd constant
    key = (key ^ 61) ^ (key >> 16);
    key = key + (key << 3);
    key = key ^ (key >> 4);
    key = key * c2;
    key = key ^ (key >> 15);
    return key;
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

// <<<1,32>>>
__global__ void inc_range(float *d_outputdist, unsigned int * d_bands_index, int start, int length, int w_index, int batch_index, int vocab_size){
  for (int i = threadIdx.x; i < length; i += blockDim.x){
      unsigned int word_index = d_bands_index[IDX2C(start + i, w_index, vocab_size)];
      atomicAdd(&d_outputdist[IDX2C(word_index, batch_index, vocab_size)], 1.0);
  }
}

// d_codes: [W, batch_size]
// d_bands_index: [vocab_size，W]
// d_outputdist: [vocab_size, batch_size]
// <<<(batch_size,batch), 256>>> : each block is responsible for each batch
// <<<64, 12>>>
template<typename dType>
__global__
void cuckoo_lookup_T(unsigned int *d_codes, dType *d_outputdist,int batch_size, int vocab_size, int W,
		     unsigned int *d_key_1, unsigned int *d_value_1, unsigned int * d_length_1,
		     unsigned int *d_key_2, unsigned int *d_value_2, unsigned int * d_length_2,
		     unsigned int *d_bands_index){
  int batch_index = blockIdx.x;
  for (int w_index = threadIdx.x; w_index < W; w_index += blockDim.x){
    unsigned int code = d_codes[w_index + batch_index * W];
    //cuckoo lookup;
    unsigned int key1 = hash_func_1_gpu(code) % vocab_size + w_index * vocab_size;
    int start = -1;
    int length = 0;
    if (d_key_1[key1] == code){
      start = d_value_1[key1];
      length = d_length_1[key1];
    } else {
      unsigned int key2 = hash_func_2_gpu(code) % vocab_size + w_index * vocab_size;
      if (d_key_2[key2] == code){
	start = d_value_2[key2];
	length = d_length_2[key2];
      }
    }

    //printf("%d %d %d\n", threadIdx.x, batch_index, length);


    if (length >= 256){
      cudaStream_t s;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      inc_range<<<1,256,0,s>>>(d_outputdist,d_bands_index,start,length,w_index,batch_index,vocab_size);
    } else {
      for (int i = 0 ; i< length; i ++ ){
	unsigned int word_index = d_bands_index[IDX2C(start + i, w_index, vocab_size)];
	atomicAdd(&d_outputdist[IDX2C(word_index, batch_index, vocab_size)], 1.0);
      }
      
    }
    
  }
}


template<typename dType>
__global__
void cuckoo_lookup_T_3(unsigned int *d_codes, dType *d_outputdist,int batch_size, int vocab_size, int W,
		     unsigned int *d_key_1, unsigned int *d_value_1, unsigned int * d_length_1,
		     unsigned int *d_key_2, unsigned int *d_value_2, unsigned int * d_length_2,
		       unsigned int *d_bands_index){
  int batch_index = blockIdx.x;
  const int maxThreads = 1024;
  __shared__ int s_w_index[maxThreads];
  __shared__ int s_start[maxThreads];
  __shared__ int s_length[maxThreads];

  for (int w_index = threadIdx.x; w_index < W; w_index += blockDim.x){
    unsigned int code = d_codes[w_index + batch_index * W];
    //cuckoo lookup;
    unsigned int key1 = hash_func_1_gpu(code) % vocab_size + w_index * vocab_size;
    int start = -1;
    int length = 0;
    if (d_key_1[key1] == code){
      start = d_value_1[key1];
      length = d_length_1[key1];
    } else {
      unsigned int key2 = hash_func_2_gpu(code) % vocab_size + w_index * vocab_size;
      if (d_key_2[key2] == code){
	start = d_value_2[key2];
	length = d_length_2[key2];
      }
    }

      s_w_index[threadIdx.x] = w_index;
      s_start[threadIdx.x] = start;
      s_length[threadIdx.x] = length; 
    
    int i_start = (threadIdx.x / 32) * 32; 
    int nalive_thread_in_warp = (blockDim.x - i_start > 32) ? 32 : blockDim.x - i_start;
    for (int i = i_start; i < i_start + 32 && i < blockDim.x; i++){
      int _length = s_length[i];
      int _w_index = s_w_index[i];
      int _start = s_start[i];

      for (int j = threadIdx.x % 32; j < _length; j += nalive_thread_in_warp){
	unsigned int word_index = d_bands_index[IDX2C(_start + j, _w_index, vocab_size)];
	atomicAdd(&d_outputdist[IDX2C(word_index, batch_index, vocab_size)], 1.0);
      } 
    }
    

  }
}

template<typename dType>
__global__
void cuckoo_lookup_T_4(unsigned int *d_codes, dType *d_outputdist,int batch_size, int vocab_size, int W,
		     unsigned int *d_key_1, unsigned int *d_value_1, unsigned int * d_length_1,
		     unsigned int *d_key_2, unsigned int *d_value_2, unsigned int * d_length_2,
		       unsigned int *d_bands_index){
  int batch_index = blockIdx.x;
  const int maxThreads = 1024;
  __shared__ int s_w_index[maxThreads];
  __shared__ int s_start[maxThreads];
  __shared__ int s_length[maxThreads];

  for (int w_index = threadIdx.x; w_index < W; w_index += blockDim.x){
    unsigned int code = d_codes[w_index + batch_index * W];
    //cuckoo lookup;
    unsigned int key1 = hash_func_1_gpu(code) % vocab_size + w_index * vocab_size;
    int start = -1;
    int length = 0;
    if (d_key_1[key1] == code){
      start = d_value_1[key1];
      length = d_length_1[key1];
    } else {
      unsigned int key2 = hash_func_2_gpu(code) % vocab_size + w_index * vocab_size;
      if (d_key_2[key2] == code){
	start = d_value_2[key2];
	length = d_length_2[key2];
      }
    }

    s_w_index[threadIdx.x] = w_index;
    s_start[threadIdx.x] = start;
    s_length[threadIdx.x] = length; 
    
    __syncthreads();
    

    int n_alive_thread = (w_index >= W / blockDim.x * blockDim.x ) ? W - W / blockDim.x * blockDim.x : blockDim.x;
    int i_start = (threadIdx.x / 32) * 32; 
    int nalive_thread_in_warp = (blockDim.x - i_start > 32) ? 32 : blockDim.x - i_start;

    int ii = threadIdx.x % 32;
    while(ii < n_alive_thread){
      int _length = atomicSub(s_length+ii, 1);
      if (_length > 0){
	int _w_index = s_w_index[ii];
	int _start = atomicAdd(s_start+ii, 1);
	unsigned int word_index = d_bands_index[IDX2C(_start, _w_index, vocab_size)];
	atomicAdd(&d_outputdist[IDX2C(word_index, batch_index, vocab_size)], 1.0);
      } else {
	ii += nalive_thread_in_warp;
      }
    }
  }
}





//<<<(batch, (vocab + 10k-1) / 10k), std::min(W,1024)>>>
template<typename dType>
__global__
void cuckoo_lookup_T_2(unsigned int *d_codes, dType *d_outputdist,int batch_size, int vocab_size, int W,
		     unsigned int *d_key_1, unsigned int *d_value_1, unsigned int * d_length_1,
		     unsigned int *d_key_2, unsigned int *d_value_2, unsigned int * d_length_2,
		       unsigned int *d_bands_index ){

  // init the shared memory to zero
  const int N = 10000;
  __shared__ int vocab_shared[N];
  int word_index_begin = blockIdx.y * N;
  int word_index_end = (word_index_begin + N > vocab_size) ? vocab_size : word_index_begin + N;
  for (int i = threadIdx.x; i < N; i += blockDim.x){
    vocab_shared[i] = 0;
  }
  __syncthreads();

  
  int batch_index = blockIdx.x;
  
  for (int w_index = threadIdx.x; w_index < W; w_index += blockDim.x){
    unsigned int code = d_codes[w_index + batch_index * W];
    //cuckoo lookup;
    unsigned int key1 = hash_func_1_gpu(code) % vocab_size + w_index * vocab_size;
    int start = -1;
    int length = 0;
    if (d_key_1[key1] == code){
      start = d_value_1[key1];
      length = d_length_1[key1];
    } else {
      unsigned int key2 = hash_func_2_gpu(code) % vocab_size + w_index * vocab_size;
      if (d_key_2[key2] == code){
	start = d_value_2[key2];
	length = d_length_2[key2];
      }
    }
    /*
    if(length < 256){
    for (int i =0; i< length; i ++){
      unsigned int word_index = d_bands_index[IDX2C(start + i, w_index, vocab_size)];
      if (word_index >= word_index_begin && word_index < word_index_end){
	atomicAdd(vocab_shared + (word_index % N), 1);	
      }
    }
    }
    */
    
  }
  
  

  //__syncthreads();

  
  /* 
  // copy the value back to global_memory
  for(int i = threadIdx.x; i < word_index_end - word_index_begin; i += blockDim.x){
    int wi = word_index_begin + i;
    d_outputdist[IDX2C(wi,batch_index, vocab_size)] = vocab_shared[i];
  }
  */
  
 
}





int main() {

  unsigned int* d_bands_index = read_matrix<unsigned int>("./data/d_bands_index_input.txt.bin");
  //50000 1000
  unsigned int* d_ht_pad_codes = read_matrix<unsigned int>("./data/d_ht_pad_codes_input.txt.bin");
  //1000 12
  unsigned int* d_key1 = read_matrix<unsigned int>("./data/d_key1_input.txt.bin");
  //50000 1000
  unsigned int* d_key2 = read_matrix<unsigned int>("./data/d_key2_input.txt.bin");
  //50000 1000
  unsigned int* d_length1 = read_matrix<unsigned int>("./data/d_length1_input.txt.bin");
  //50000 1000
  unsigned int* d_length2 = read_matrix<unsigned int>("./data/d_length2_input.txt.bin");
  //50000 1000
  float* d_dist_input = read_matrix<float>("./data/d_outputdist_lookup_input.txt.bin");
  //50000 12
  unsigned int* d_value1 = read_matrix<unsigned int>("./data/d_value1_input.txt.bin");
  //50000 1000
  unsigned int* d_value2 = read_matrix<unsigned int>("./data/d_value2_input.txt.bin");
  //50000 1000

  // output
  float* d_dist_output = read_matrix<float>("./data/d_outputdist_input.txt.bin");

  checkCudaError(cudaMalloc(&d_starts, 1024 * batch*sizeof(unsigned int)));
  checkCudaError(cudaMalloc(&d_lengths, 1024 * batch*sizeof(unsigned int)));



  { //dense2array

    float msecTotal = 0.0f;
    int Test = 1;
    
    int *d_starts, *d_lengths; 
    cudaMalloc((void **)&d_starts, W * batch * sizeof(int));
    cudaMalloc((void **)&d_lengths, W * batch * sizeof(int));

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    
    checkCudaError(cudaEventRecord(start, NULL));

    for (int i = 0; i < Test; i++) {
      cudaMemset(d_dist_input, 0, voc * batch* sizeof(float)); //0.02ms

      cuckoo_lookup_T_3<<<batch, std::min(1024,W)>>>(d_ht_pad_codes, d_dist_input, batch, voc, W, d_key1, d_value1, d_length1, d_key2, d_value2, d_length2, d_bands_index); // 0.9 ms

      cudaDeviceSynchronize();


      //cuckoo_lookup_T<<<batch, std::min(1024,W)>>>(d_ht_pad_codes, d_dist_input, batch, voc, W, d_key1, d_value1, d_length1, d_key2, d_value2, d_length2, d_bands_index); // 0.9 ms


      //const int N = 10000;
      //cuckoo_lookup_T_2<<<dim3(batch, (voc+N-1)/N ), std::min(1024,W)>>>(d_ht_pad_codes, d_dist_input, batch, voc, W, d_key1, d_value1, d_length1, d_key2, d_value2, d_length2, d_bands_index); // 0.9 ms 
    }
    
    checkCudaError(cudaEventRecord(stop, NULL));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));
    msecTotal /= Test;
    LOG(INFO)  << "cuckoo_lookup Time= " << msecTotal << " msec, ";
    checkCudaError(cudaGetLastError());

    int offset = 0;

    std::cout << "d_dist_input\n";
    print_matrix_gpu(d_dist_input, voc, batch, offset, offset+10, 0, batch);

    std::cout << "d_dist_output\n";
    print_matrix_gpu(d_dist_output, voc, batch, offset, offset+10, 0, batch);



    /*
    std::cout << "d_bands_index\n";
    print_matrix_gpu(d_bands_index, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_ht_pad_codes\n";
    print_matrix_gpu(d_ht_pad_codes, W, batch, offset, offset+10, 0, batch);

    std::cout << "d_key1\n";
    print_matrix_gpu(d_key1, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_value1\n";
    print_matrix_gpu(d_value1, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_length1\n";
    print_matrix_gpu(d_length1, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_key2\n";
    print_matrix_gpu(d_key2, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_value2\n";
    print_matrix_gpu(d_value2, voc, W, offset, offset+10, 0, batch);

    std::cout << "d_length2\n";
    print_matrix_gpu(d_length2, voc, W, offset, offset+10, 0, batch);

    */

  }

  return 0;
}
