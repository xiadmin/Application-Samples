CXX=g++
CXXLANG= -std=c++17
CXXFLAGS= -Wpedantic -O0 -g -pg

SOURCES=main.cpp depacking_cpu.cpp
OBJECTS=$(SOURCES:.cpp=.o) 

CUDA_CXX=nvcc
CUDA_CXXFLAGS= -g -G -pg -Xcompiler -O0 -Xptxas -O0

CUDA_SOURCES=depacking.cu
CUDA_OBJECTS=$(CUDA_SOURCES:.cu=.o) 


INCLUDE_DIRS= -I/usr/include/ -I/usr/local/include 
INCLUDE_DIRS+= -I/usr/local/cuda/include -I/usr/local/cuda/targets/aarch64-linux/include/  `pkg-config --cflags opencv4`

LDLIBS= -lm3api -lcudart -lnppc -lnppicc `pkg-config --libs opencv4` 

LDFLAGS= -L/usr/lib -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/aarch64-linux/lib

PROGRAM=jetson_sample

.PHONY: all clean

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXLANG) $(OBJECTS) $(CUDA_OBJECTS) -o $(PROGRAM) $(LDFLAGS) $(LDLIBS)

$(OBJECTS): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(CXXLANG) -c $(SOURCES) $(INCLUDE_DIRS) 

$(CUDA_OBJECTS): $(CUDA_SOURCES)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(CXXLANG) -c $(CUDA_SOURCES) $(INCLUDE_DIRS) 

clean:
	rm -f *.o *~ $(PROGRAM) $(OBJECTS)         

#Feel free to enable code optimalizations by changing the -O flags, but disable debugging (-g -G) first

#For reference on makefiles, see
#https://www.gnu.org/software/make/manual/make.html

#For reference on compilation with g++, see
#https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html

