# Makefile for MPI components
# Supports compiling C benchmarks and C++ LibTorch inference engine

CC = mpicc
CXX = mpicxx
CFLAGS = -O3 -march=native -Wall
CXXFLAGS = -O3 -std=c++17 -Wall

# Paths for LibTorch and OpenCV (adjust LIBTORCH_PATH as needed on master/workers)
LIBTORCH_PATH ?= /usr/local/libtorch
OPENCV_FLAGS ?= $(shell pkg-config --cflags --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_videoio -lopencv_imgproc")

TORCH_INCLUDES = -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
TORCH_LIBS = -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH_PATH)/lib

TARGETS = hello_mpi mpi_trap_pi matmul_benchmark

.PHONY: all clean benchmarks

all: benchmarks

benchmarks: $(TARGETS)

hello_mpi: hello_mpi.c
	$(CC) $(CFLAGS) -o $@ $<

mpi_trap_pi: mpi_trap_pi.c
	$(CC) $(CFLAGS) -o $@ $<

matmul_benchmark: matmul_benchmark.c
	$(CC) $(CFLAGS) -o $@ $<

# Build LibTorch + OpenMPI node (requires LibTorch and OpenCV)
violence_mpi_master: violence_mpi_master.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCLUDES) -o $@ $< $(OPENCV_FLAGS) $(TORCH_LIBS)

clean:
	rm -f $(TARGETS) violence_mpi_master
