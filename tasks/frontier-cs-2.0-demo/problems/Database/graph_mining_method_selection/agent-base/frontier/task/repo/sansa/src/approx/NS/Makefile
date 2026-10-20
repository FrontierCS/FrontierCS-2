include ../../common.mk
OBJS := VertexSet.o graph.o path_sampler.o
VPATH += ../../common
BIN=../../../bin/
INCLUDES = -I../../../include

all: ns_path ns_clique ns_cycle ns_diamond ns_house ns_timer ns_dumbbell ns_all ns_online ns_online_transparent ns_final ns_profile ns_online_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

ns_path_gpu: $(OBJS) ns_path_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_path_gpu.o -o $@ -lgomp -lcudart -lcurand
	mv $@ $(BIN)

ns_path: $(OBJS) ns_path.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_path.o -o $@ -lgomp
	mv $@ $(BIN)

ns_clique_gpu: $(OBJS) ns_clique_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_clique_gpu.o -o $@ -lgomp -lcudart -lcurand
	mv $@ $(BIN)

ns_clique: $(OBJS) ns_clique.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_clique.o -o $@ -lgomp
	mv $@ $(BIN)

ns_diamond: $(OBJS) ns_diamond.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_diamond.o -o $@ -lgomp
	mv $@ $(BIN)

ns_tailedtri: $(OBJS) ns_tailedtri.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_tailedtri.o -o $@ -lgomp
	mv $@ $(BIN)

ns_star: $(OBJS) ns_star.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_star.o -o $@ -lgomp
	mv $@ $(BIN)

ns_timer: $(OBJS) ns_timer.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_timer.o -o $@ -lgomp
	mv $@ $(BIN)

ns_cycle: $(OBJS) ns_cycle.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_cycle.o -o $@ -lgomp
	mv $@ $(BIN)

ns_house: $(OBJS) ns_house.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_house.o -o $@ -lgomp
	mv $@ $(BIN)

ns_house_gpu: $(OBJS) ns_house_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_house_gpu.o -o $@ -lgomp -lcudart -lcurand
	mv $@ $(BIN)

ns_dumbbell: $(OBJS) ns_dumbbell.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_dumbbell.o -o $@ -lgomp
	mv $@ $(BIN)

ns_all: $(OBJS) ns_all.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_all.o -o $@ -lgomp
	mv $@ $(BIN)

ns_online: $(OBJS) ns_online.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_online.o -o $@ -lgomp
	mv $@ $(BIN)

ns_online_transparent: $(OBJS) ns_online_transparent.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_online_transparent.o -o $@ -lgomp
	mv $@ $(BIN)

ns_final: $(OBJS) ns_final.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_final.o -o $@ -lgomp
	mv $@ $(BIN)

ns_online_base: $(OBJS) ns_online_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_final.o -o $@ -lgomp
	mv $@ $(BIN)

ns_profile: $(OBJS) ns_profile.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_profile.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
