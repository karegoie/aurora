# Aurora Gene Predictor Makefile
# Compiles both C and C++ sources with required libraries

CC = gcc
CXX = g++
CFLAGS = -O3 -Wall -std=c11 -fopenmp
CXXFLAGS = -O3 -Wall -std=c++17 -fopenmp
LDFLAGS = -fopenmp

# Library paths and flags
# Note: Adjust these paths based on your system
FFTW_LIBS = -lfftw3 -lfftw3_omp -lm
TORCH_PREFIX = $(PWD)/libtorch
TORCH_LIB_DIR = $(TORCH_PREFIX)/lib
# Use dynamic linking for libtorch by default (shared libs in libtorch/lib)
TORCH_LIBS = -L$(TORCH_LIB_DIR) -Wl,-rpath,$(TORCH_LIB_DIR) -ltorch -ltorch_cpu -ltorch_global_deps -lc10
TORCH_INCLUDES = -I$(TORCH_PREFIX)/include -I$(TORCH_PREFIX)/include/torch/csrc/api/include

# All libraries
LIBS = $(FFTW_LIBS) $(TORCH_LIBS) -lstdc++ -lpthread -ldl
INCLUDES = -Isrc -Iinclude $(TORCH_INCLUDES)

# Source files
C_SOURCES = src/main.c src/config.c src/fasta.c src/utils.c src/cwt.c \
            src/model.c src/train.c src/predict.c
CXX_SOURCES = src/rl_agent.cpp

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)
ALL_OBJECTS = $(C_OBJECTS) $(CXX_OBJECTS)

# Target executable
TARGET = aurora

# Default target
all: $(TARGET)

# Link all object files
$(TARGET): $(ALL_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(ALL_OBJECTS) $(TARGET)
	rm -f src/*.o

# Phony targets
.PHONY: all clean

# Static build target: attempt to produce a statically-linked binary. Note:
# static linking may fail if dependent libraries (e.g. libtorch) are only
# available as shared objects. Use this target as a best-effort option.
# (No static targets) Build uses dynamic linking for external libs by default.

# Dependencies (simplified - in production, use makedepend or similar)
src/main.o: src/main.c src/aurora.h src/config.h src/train.h src/predict.h
src/config.o: src/config.c src/config.h src/aurora.h
src/fasta.o: src/fasta.c src/fasta.h src/aurora.h
src/utils.o: src/utils.c src/utils.h src/aurora.h
src/cwt.o: src/cwt.c src/cwt.h src/aurora.h
src/model.o: src/model.c src/model.h src/aurora.h
src/train.o: src/train.c src/train.h src/aurora.h src/fasta.h src/cwt.h src/rl_agent.h
src/predict.o: src/predict.c src/predict.h src/aurora.h src/fasta.h src/cwt.h src/rl_agent.h
src/rl_agent.o: src/rl_agent.cpp src/rl_agent.h src/aurora.h
