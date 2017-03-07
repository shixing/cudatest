CC=g++
NVCC=nvcc
FLAGS=-O2
NVFLAGS=$(FLAGS) -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
LDFLAGS=-lcudart -lcublas

main: main.cuo
	$(CC) -o $@ $< $(LDFLAGS)

%.o: %.cc
	$(CC) -o $@ -c $< 
%.cuo: %.cu
	$(NVCC) $(NVFLAGS) -o $@ -c $< 

clean:
	rm -rf *.o *.cuo main
