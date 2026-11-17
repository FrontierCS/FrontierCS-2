include ../common.mk
#VPATH += ../partitioner
all: ASAP-path 

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

ASAP-path: graph.o VertexSet.o ASAP-path.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) graph.o VertexSet.o ASAP-path.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
