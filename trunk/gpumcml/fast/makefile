#
# Makefile for GPUMCML on Linux
#

######################################################################
# Configurations
######################################################################

# To disable register usage constraint, set it to 0.
MAX_REG_COUNT_SM20 := 36
MAX_REG_COUNT_SM12 := 0

# To compile 32-bit binaries on a 64-bit machine, set this to 1.
COMPILE_32 := 1

######################################################################
# Compiler flags
######################################################################

CUDA_INSTALL_ROOT ?= /usr/local/cuda

NVCC := $(CUDA_INSTALL_ROOT)/bin/nvcc
GCC := g++

HP_64 = $(shell uname -m | grep 64)
ifeq "$(strip $(HP_64))" ""
  # 32-bit architecture
	LIBCUDART_DIR := -L$(CUDA_INSTALL_ROOT)/lib
	LIBCUTIL_NAME := cutil_i386
else
  ifeq "$(strip $(COMPILE_32))" "1"
    # 64-bit architecture, but we compile 32-bit binary.
	  LIBCUDART_DIR := -L$(CUDA_INSTALL_ROOT)/lib
	  LIBCUTIL_NAME := cutil_i386
	else
	  LIBCUDART_DIR := -L$(CUDA_INSTALL_ROOT)/lib64
	  LIBCUTIL_NAME := cutil_x86_64
  endif
endif

CFLAGS := -Wall -O3
LDFLAGS := $(LIBCUDART_DIR) -Lcutil-linux

NVCC_FLAGS := -I$(CUDA_INSTALL_ROOT)/include -Icutil-linux
NVCC_FLAGS += -O3 -DUNIX -use_fast_math #-keep # -deviceemu
NVCC_FLAGS += -Xptxas -v
#export PTX_FLAGS := $(PTXAS_FLAGS) -maxrregcount 64 -mem -v

ifeq "$(strip $(COMPILE_32))" "1"
  CFLAGS += -m32
  LDFLAGS += -m32
  NVCC_FLAGS += -m32
endif

######################################################################
# Targets
######################################################################

PROG_BASE := gpumcml
PROG_SM20 := $(PROG_BASE).sm_20
PROG_SM12 := $(PROG_BASE).sm_12

default: $(PROG_SM20) $(PROG_SM12)

######################################################################
# GPU code compilation rules for Compute Capability 2.0
######################################################################

GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
NVCC_FLAGS_SM20 := $(NVCC_FLAGS) $(GENCODE_SM20) -DCUDA_ARCH=20
ifneq "$(strip $(MAX_REG_COUNT_SM20))" "0"
NVCC_FLAGS_SM20 += -maxrregcount $(MAX_REG_COUNT_SM20)
endif

$(PROG_SM20): $(PROG_SM20).cu_o gpumcml_io.o
	$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM20).cu_o: $(PROG_BASE)_main.cu
	$(NVCC) $(NVCC_FLAGS_SM20) -o $@ -c $<

clean_sm_20:
	$(NVCC) $(NVCC_FLAGS_SM20) -o $(PROG_SM20).cu_o -c $(PROG_BASE)_main.cu -clean

######################################################################
# GPU code compilation rules for Compute Capability 1.2
######################################################################

GENCODE_SM12 := -gencode=arch=compute_12,code=\"sm_12,compute_12\"
NVCC_FLAGS_SM12 := $(NVCC_FLAGS) $(GENCODE_SM12) $(GENCODE_SM20) -DCUDA_ARCH=12
ifneq "$(strip $(MAX_REG_COUNT_SM12))" "0"
NVCC_FLAGS_SM12 += -maxrregcount $(MAX_REG_COUNT_SM12)
endif

$(PROG_SM12): $(PROG_SM12).cu_o gpumcml_io.o
	$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM12).cu_o: $(PROG_BASE)_main.cu
	$(NVCC) $(NVCC_FLAGS_SM12) -o $@ -c $<

clean_sm_12:
	$(NVCC) $(NVCC_FLAGS_SM12) -o $(PROG_SM20).cu_o -c $(PROG_BASE)_main.cu -clean

# Disable implicit rules on .cu files.
%.cu: ;

######################################################################
# C code compilation rules
######################################################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $<

%.o: %.cpp
	$(GCC) $(CFLAGS) -c $<

######################################################################

clean: clean_sm_20 clean_sm_12
	rm -f $(PROG_SM20) $(PROG_SM12) *.o *.cu_o

######################################################################

.PHONY: default clean clean_sm_20 clean_sm_12

