#nvcc -lcublas -lcusparse  -gencode arch=compute_35,code=sm_35 main.cu -o main
source /usr/usc/gnu/gcc/4.9.3/setup.sh

nvcc -O3 -arch=sm_35 -rdc=true -std=c++11 main.cu -lcudadevrt -lcublas -o main
