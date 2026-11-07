SRCS=$(wildcard ./src/*.cpp)
OBJDIR=./build
RESDIR=./results
OBJS = $(patsubst ./src/%.cpp,$(OBJDIR)/%.o,$(SRCS))
TARGET = fg
CXX := g++
CXXFLAGS := -std=c++11 -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0


all: $(TARGET)

$(OBJDIR)/%.o:src/%.cpp
	@test -d $(OBJDIR) | mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET):$(OBJS)
	@test -d $(RESDIR) | mkdir -p $(RESDIR)
	$(CXX) $(CXXFLAGS)  -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJDIR)
