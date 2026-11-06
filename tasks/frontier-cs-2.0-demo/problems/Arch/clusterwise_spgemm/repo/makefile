RMATPATH = GTgraph/R-MAT
SPRNPATH = GTgraph/sprng2.0-lite

include GTgraph/Makefile.var
INCLUDE = -I/opt/intel/tbb/include
INCLUDE += -I$(SPRNPATH)/include
COMPILER = CC

FLAGS = -g -dynamic -qopenmp -O3 -ltbb -ltbbmalloc
FLAGS += -mkl
PRINT_METADATA_TO_FILE_FLAG = -DPRINT_METADATA_TO_FILE

ifeq ($(PRINT_METADATA_TO_FILE), 1)
    FLAGS += $(PRINT_METADATA_TO_FILE_FLAG)
endif

#-lnuma -lmemkind -lpthread
#-qopt-report=5 -xMIC-AVX512 

sprng:	
	(cd $(SPRNPATH); $(MAKE); cd ../..)

rmat:	sprng
	(cd $(RMATPATH); $(MAKE); cd ../..)

TOCOMPILE = $(RMATPATH)/graph.o $(RMATPATH)/utils.o $(RMATPATH)/init.o $(RMATPATH)/globals.o 

# flags defined in GTgraph/Makefile.var
SAMPLE = ./sample
BIN = ./bin
SRC_SAMPLE = $(wildcard $(SAMPLE)/*.cpp)
SAMPLE_TARGET = $(SRC_SAMPLE:$(SAMPLE)%=$(BIN)%)

sample_knl: rmat $(SAMPLE_TARGET:.cpp=_knl)
sample_hw: rmat $(SAMPLE_TARGET:.cpp=_hw)

$(BIN)/%_knl: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(COMPILER) $(FLAGS) $(INCLUDE) -o $@ $^ -DTBB -DKNL_EXE ${TOCOMPILE} ${LIBS}

$(BIN)/%_hw: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(COMPILER) $(FLAGS) $(INCLUDE) -o $@ $^ -DTBB -DHW_EXE ${TOCOMPILE} ${LIBS}

clean:
	(cd GTgraph; make clean; cd ../..)
	rm -rf ./bin/*
