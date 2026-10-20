CXX := g++
CXXFLAGS := -std=c++11 -O3 -DNDEBUG -fopenmp
SOURCES := src/alg.cpp src/basis.cpp src/Preprocess.cpp src/mf_alsh.cpp src/main.cpp
all: fg
fg: $(SOURCES) $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@
clean:
	rm -f fg
.PHONY: all clean
