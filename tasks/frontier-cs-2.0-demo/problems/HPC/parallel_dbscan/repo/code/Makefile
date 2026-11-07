include parallelDefs

GRIDS = Grids.h GridKDTree.h GetNeighborKDTree.h GetNeighborTypeDef.h
BENCH_REQUIRE = $(GRIDS) unionFind.h DBSCANGeometry.h je_allocator.h
GLOBAL_REQUIRE = parallel.h sequence.h quickSort.h ndHash.h sampleSort.h transpose.h blockRadixSort.h
TOPOLOGY = topology.h deterministicHash.h
COMMON = IO.h parseCommandLine.h utils.h geometry.h geometryIO.h gettime.h
TIME_GLOBAL_REQUIRE =
TIME_REQUIRE = DBSCAN.h
OBJS = DBSCANTime.o DBSCAN.o GetNeighborKDTree.o Grids.o

EVERYTHING = $(GRIDS) $(BENCH_REQUIRE) $(GLOBAL_REQUIRE) $(TOPOLOGY) $(COMMON) $(TIME_REQUIRE) 

ifdef USE_JEMALLOC

%.o : %.C $(EVERYTHING)
	$(PCC) $(PCFLAGS) -c $< -o $@ -L`jemalloc-config --libdir` -Wl,-rpath,`jemalloc-config --libdir` -ljemalloc `jemalloc-config --libs`

DBSCAN: $(OBJS)
	$(PCC) $(PLFLAGS) -o $@ $(OBJS) -L`jemalloc-config --libdir` -Wl,-rpath,`jemalloc-config --libdir` -ljemalloc `jemalloc-config --libs`

else

%.o : %.C $(EVERYTHING)
	$(PCC) $(PCFLAGS) -c $< -o $@

DBSCAN: $(OBJS)
	$(PCC) $(PLFLAGS) -o $@ $(OBJS)

endif

clean :
	rm -f $(BNCHMRK) *.o *.pyc;

cleansrc :
	make -s clean
	rm -f $(GLOBAL) $(GLOBAL_LIB_REQUIRE) $(BENCH) $(TEST_FILES)


