include ../../common.mk
OBJS := VertexSet.o graph.o
VPATH += ../../common
BIN=../../../bin
INCLUDES = -I../../../include
INCLUDES+=-I./exact-counting

all: ns_fast_profiler

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

ns_fast_profiler: $(OBJS) ns_fast_profiler.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_fast_profiler.o -o $@ -lgomp
	mv $@ $(BIN)


clean:
	rm *.o
