CC = clang
TARGET :=simple
SRC := simpleTest
OBJ := $(SRC:.cpp=.o)
UNAME := $(shell uname)

OS_TYPE := Parallel Linux
PFLAG := -DPARALLEL -D_BSD_SOURCE -D_XOPEN_SOURCE=700
# -g needed for test framework assertions
#
# switch -fdetach to -ftapir to enable Tapir optimization passes for better
# performance.
CFLAGS := -std=c++14 -Wall -g
LDFLAGS= -Wall -lm -lrt -ldl -lpthread -lopenmp -fuse-ld=gold


ifeq ($(DEBUG),1)
	CFLAGS += -O0 -DDEBUG $(PFLAG)
else
	CFLAGS += -O3 -flto -DNDEBUG -march=native
endif

ifeq ($(REFERENCE),1)
	CFLAGS += -DRUN_REFERENCE_CODE=1
endif

CFLAGS += $(OTHER_CFLAGS)

LDFLAGS= -Wall -lm -lrt -ldl -flto -lpthread 
ifeq ($(PROFILE),1)
	CFLAGS += -DPROFILE
	LDFLAGS += -lprofiler
endif


COMMON = common_sort.h regionsSort.h parallel.h prefixsum.h radix_configs.h sequence.h sequenceIO.h triangle.h utils.h IO.h edge.h edgelistgraph.h block.h example.cpp misc.h radixSort.h

.PHONY : default clean



default : radixSort radixSort_cycle 

radixSort : $(COMMON)
	g++ -I. -std=c++14 example.cpp -DCILKP -DBITS_HACK -O3 -flto -march=native -fcilkplus -lcilkrts -o radixSort

radixSort_cycle : $(COMMON)
	g++ -I. -std=c++14 example.cpp -DCILKP -DCYCLE -DBITS_HACK -O3 -flto -march=native -fcilkplus -lcilkrts -o radixSort_cycle

# Each C source file will have a corresponding file of prerequisites.
# Include the prerequisites for each of our C source files.
-include $(SRC:.cpp=.d)

# This rule generates a file of dependencies (i.e., a makefile) %.d
# from the C source file %.c.
%.d : %.cpp
	@set -e; rm -f $@; \
	$(CC) -MM -MT $*.o -MP -MF $@.$$$$ $(CFLAGS) $<; \
	sed -e 's|\($*\)\.o[ :]*|\1.o $@ : |g' < $@.$$$$ > $@; \
	rm -f $@.$$$$*

# We could use the -MMD and -MP flags here to have this rule generate
# the dependencies file.
%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

simple : simple.o $(OBJ)
	$(CC) $^ $(LDFLAGS) -o $@ -lrt

clean :
	rm -f *.o *.d* *~ $(TARGET) radixSort radixSort_cycle  



