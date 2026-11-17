include ../../common.mk
VPATH += ../../common
all: sansa_omp_base
BIN=../../../bin
INCLUDES = -I../../../include
INCLUDES+=-I./exact-counting


clean:
	rm *.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

# baseline version, add make USE_SET_OPS=1 when make
sansa_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)