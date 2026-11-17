#include "graph.h"
#include <cmath>


class Sampler {
    public:
        virtual void GetGraph(Graph& g, float delta, Pattern& p) {}
        virtual uint64_t Scale(uint64_t count) {return count * scale_factor;}
        uint64_t scale_factor;

};

// Derived class
class ColorSampler: public Sampler{
   public:
        ColorSampler() {}
        uint64_t Scale(uint64_t count) {return count * scale_factor;}
        
        void GetGraph(Graph& g, float delta, Pattern& p) {
            if(delta != 1) {
                g.color_sparsify_fast(delta); 
            }
            scale_factor = std::pow(delta, p.get_nodes()-1);
        }
};