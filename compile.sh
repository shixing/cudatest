nvcc -lcublas -lcusparse  -gencode arch=compute_35,code=sm_35 main.cu -o main
