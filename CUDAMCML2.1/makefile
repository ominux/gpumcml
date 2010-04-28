
NVCC=nvcc
GCC=g++

#export PTX_FLAGS := $(PTXAS_FLAGS) -maxrregcount 64 -mem -v

CFLAGS=-Wall -O3
LDFLAGS=-L/usr/local/cuda/lib -Lcutil-linux

NVCC_FLAGS+=-O5 -DUNIX -arch sm_13 -use_fast_math # -keep -deviceemu
NVCC_FLAGS+= -I/usr/local/cuda/include -Icutil-linux

######################################################################

PROG_CUDA=cudamcml
default: clean $(PROG_CUDA)

######################################################################

$(PROG_CUDA): $(PROG_CUDA).cu_o cudamcml_io.o
	$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -lcutil

$(PROG_CUDA).cu_o: $(PROG_CUDA)_main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

######################################################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $<

%.o: %.cpp
	$(GCC) $(CFLAGS) -c $<

######################################################################

clean:
	rm -f $(PROG_CUDA) *.o *.cu_o
	$(NVCC) $(NVCC_FLAGS) -o $(PROG_CUDA).cu_o -c $(PROG_CUDA)_main.cu -clean

######################################################################

.PHONY: default clean

