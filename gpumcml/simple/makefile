#
# Makefile for GPUMCML on Linux
#
# It generates four executables: gpumcml.sm_[11 | 12 | 13 | 20].
#
# They are optimized for GPUs with compute capability 1.1, 1.2, 1.3 and 2.0
# respectively. Note that a program can run on a GPU with compute capability
# that is higher than the one it is optimized for.
#

######################################################################
# Configurations
######################################################################

# To disable register usage constraint, set it to 0.
MAX_REG_COUNT_SM20 := 0
MAX_REG_COUNT_SM13 := 0
MAX_REG_COUNT_SM12 := 0
MAX_REG_COUNT_SM11 := 0

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

OS_NAME = $(shell uname -s)
ifeq "$(OS_NAME)" "Linux"
  # Linux OS
  LIBCUTIL_DIR := cutil-linux
else
  # OS X
  LIBCUTIL_DIR := cutil-osx
endif

CFLAGS := -Wall -O3
LDFLAGS := $(LIBCUDART_DIR) -L$(LIBCUTIL_DIR)

NVCC_FLAGS := -I$(CUDA_INSTALL_ROOT)/include -I$(LIBCUTIL_DIR)
NVCC_FLAGS += -O3 -DUNIX -use_fast_math
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
PROG_SM13 := $(PROG_BASE).sm_13
PROG_SM12 := $(PROG_BASE).sm_12
PROG_SM11 := $(PROG_BASE).sm_11

default: $(PROG_SM20) $(PROG_SM13) $(PROG_SM12) $(PROG_SM11)

######################################################################
# Source files
######################################################################

CU_SRCS := $(PROG_BASE)_main.cu \
		$(PROG_BASE)_kernel.cu	\
		$(PROG_BASE)_mem.cu \
		$(PROG_BASE)_rng.cu \
		$(PROG_BASE)_kernel.h	\
		$(PROG_BASE).h

######################################################################
# GPU code compilation rules for Compute Capability 2.0
######################################################################

GENCODE_SM20 := -gencode arch=compute_20,code=\"sm_20,compute_20\"
NVCC_FLAGS_SM20 := $(NVCC_FLAGS) $(GENCODE_SM20) -D__CUDA_ARCH__=200
ifneq "$(strip $(MAX_REG_COUNT_SM20))" "0"
NVCC_FLAGS_SM20 += -maxrregcount $(MAX_REG_COUNT_SM20)
endif
NVCC_FLAGS_SM20 += -keep # -deviceemu

$(PROG_SM20): $(PROG_SM20).cu_o gpumcml_io.o
	-$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM20).cu_o: $(CU_SRCS)
	-$(NVCC) $(NVCC_FLAGS_SM20) -o $@ -c $<

clean_sm_20:
	-$(NVCC) $(NVCC_FLAGS_SM20) -o $(PROG_SM20).cu_o -c $(PROG_BASE)_main.cu -clean

######################################################################
# GPU code compilation rules for Compute Capability 1.3
######################################################################

GENCODE_SM13 := -gencode arch=compute_13,code=\"sm_13,compute_13\"
NVCC_FLAGS_SM13 := $(NVCC_FLAGS) $(GENCODE_SM13) -D__CUDA_ARCH__=130
ifneq "$(strip $(MAX_REG_COUNT_SM13))" "0"
NVCC_FLAGS_SM13 += -maxrregcount $(MAX_REG_COUNT_SM13)
endif
NVCC_FLAGS_SM13 += #-keep # -deviceemu

$(PROG_SM13): $(PROG_SM13).cu_o gpumcml_io.o
	-$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM13).cu_o: $(CU_SRCS)
	-$(NVCC) $(NVCC_FLAGS_SM13) -o $@ -c $<

clean_sm_13:
	-$(NVCC) $(NVCC_FLAGS_SM13) -o $(PROG_SM13).cu_o -c $(PROG_BASE)_main.cu -clean

######################################################################
# GPU code compilation rules for Compute Capability 1.2
######################################################################

GENCODE_SM12 := -gencode arch=compute_12,code=\"sm_12,compute_12\"
NVCC_FLAGS_SM12 := $(NVCC_FLAGS) $(GENCODE_SM12) -D__CUDA_ARCH__=120
ifneq "$(strip $(MAX_REG_COUNT_SM12))" "0"
NVCC_FLAGS_SM12 += -maxrregcount $(MAX_REG_COUNT_SM12)
endif
NVCC_FLAGS_SM12 += #-keep # -deviceemu

$(PROG_SM12): $(PROG_SM12).cu_o gpumcml_io.o
	-$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM12).cu_o: $(CU_SRCS)
	-$(NVCC) $(NVCC_FLAGS_SM12) -o $@ -c $<

clean_sm_12:
	-$(NVCC) $(NVCC_FLAGS_SM12) -o $(PROG_SM12).cu_o -c $(PROG_BASE)_main.cu -clean

######################################################################
# GPU code compilation rules for Compute Capability 1.1
######################################################################

GENCODE_SM11 := -gencode arch=compute_11,code=\"sm_11,compute_11\"
NVCC_FLAGS_SM11 := $(NVCC_FLAGS) $(GENCODE_SM11) -D__CUDA_ARCH__=110
ifneq "$(strip $(MAX_REG_COUNT_SM11))" "0"
NVCC_FLAGS_SM11 += -maxrregcount $(MAX_REG_COUNT_SM11)
endif
NVCC_FLAGS_SM11 += #-keep # -deviceemu

$(PROG_SM11): $(PROG_SM11).cu_o gpumcml_io.o
	-$(GCC) $(LDFLAGS) -o $@ $^ -lcudart -l$(LIBCUTIL_NAME)

$(PROG_SM11).cu_o: $(CU_SRCS)
	-$(NVCC) $(NVCC_FLAGS_SM11) -o $@ -c $<

clean_sm_11:
	-$(NVCC) $(NVCC_FLAGS_SM11) -o $(PROG_SM11).cu_o -c $(PROG_BASE)_main.cu -clean

######################################################################
# C code compilation rules
######################################################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $<

%.o: %.cpp
	$(GCC) $(CFLAGS) -c $<

######################################################################

clean: clean_sm_20 clean_sm_13 clean_sm_12 clean_sm_11
	rm -f $(PROG_SM20) $(PROG_SM13) $(PROG_SM12) $(PROG_SM11) *.o *.cu_o

######################################################################

# Disable implicit rules on .cu files.
%.cu: ;

.PHONY: default clean clean_sm_20 clean_sm_13 clean_sm_12 clean_sm_11

