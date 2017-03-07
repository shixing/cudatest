#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <glog/logging.h>



#define checkCudaError(stmt)                        \
    do {                                            \
      cudaError_t status = (stmt);                  \
      if (status != cudaSuccess) {                  \
        LOG(FATAL) << "CUDA failure: "              \
                   << cudaGetErrorString(status);   \
      }                                             \
    }while(0)

#define checkCublasError(stmt)                      \
    do {                                            \
      cublasStatus_t status = (stmt);               \
      if (status != CUBLAS_STATUS_SUCCESS) {        \
        LOG(FATAL) << "CUDA failure: "              \
                   << status;                       \
      }                                             \
    }while(0)

