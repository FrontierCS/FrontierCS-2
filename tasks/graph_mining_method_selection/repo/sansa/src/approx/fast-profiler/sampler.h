#include "graph.h"
#include "scan.h"
#include <cmath>
#include <random>



class Sampler {
    public:
        virtual void GetGraph(Graph& g, Graph* subgraph, float delta, Pattern& p) {}
        virtual uint64_t Scale(uint64_t count) {return count * scale_factor;}
        uint64_t scale_factor;

};

// Derived class
// class MultiColorSampler: public Sampler{
//    public:
//         MultiColorSampler() {}
//         uint64_t Scale(uint64_t count) {return count * scale_factor;}
        
//         void GetMultiGraph(Graph& g, Graph** subgraph, int num_colorings, float delta, Pattern& p) {
            
//             printf("sf: %ld\n", scale_factor);
//             int c = (int) delta;

//             if(delta == 1) {
//                 return;
//             }
//             scale_factor = std::pow(delta, p.get_nodes()-1);

//             Timer t;
//             t.Start();
//             const int v = g.V();
//             auto colors = new int[num_colorings][v];

//             std::vector<vidType> new_degrees(g.V(), 0);

//             std::random_device rd;
//             std::mt19937 mt(rd());
//             std::uniform_int_distribution<int> dist(0, c-1);

//             //#pragma omp parallel for schedule(dynamic, 1) shared(colors)
//             for (vidType v = 0; v < g.V(); v++) {
//                 for (int i = 0; i < num_colorings; i++) {
//                     colors[i][v] = dist(mt);
//                 }
//             }

//             t.Stop();
//             std::cout << "coloring = " << t.Seconds() << " sec\n";

//             t.Start();

//             #pragma omp parallel for shared(colors, new_degrees)
//             for (vidType src = 0; src < g.V(); src ++) {  
//                 for(auto dst : g.N(src)) {
//                     for(int i = 0; i < num_colorings; i++) {
//                         if(colors[i][src] == colors[i][dst]) { //keep edge
//                             new_degrees[i][src] += 1;
//                         }
//                     } 
//                 }
//             }


//             auto new_vertices =  new eidType*[num_colorings];
           

//             //auto new_edges = new vidType[num_edges];

//             for(int i = 0; i < num_colorings; i++) {
//                 new_vertices[i] = custom_alloc_global<eidType>(g.V()+1);
//                 parallel_prefix_sum<vidType,eidType>(new_degrees[i], new_vertices[i]);
//                 auto num_edges = new_vertices[i][g.V()];
//                 subgraph[i]->allocateFrom(g.V(), num_edges);
//             }

//             #pragma omp parallel for
//             for (vidType src = 0; src < g.V(); src ++) {
//                 auto offset = new eidType[num_colorings];
//                 auto begin = new vidType[num_colorings];
                
//                 for(int i = 0; i < num_colorings; i++) {
//                     begin[i] = new_vertices[i][src];
//                     subgraph[i]->fixEndEdge(src, new_vertices[i][src+1]); // fix row pointers
//                     offset[i] = 0;
//                 }
                
//                 for (auto dst : g.N(src)) {
//                     for(int i = 0; i < num_colorings; i++) {
//                         if (colors[i][src] == colors[i][dst]) { // keep edge
//                             // new_edges[begin+offset] = dst;
//                             subgraph[i]->constructEdge(begin[i]+offset[i], dst);
//                             offset[i] ++;
//                         }
//                     }
//                 }
//             }

        
//             t.Stop();
//             std::cout << "color induced sparsification fast = " << t.Seconds() << " sec\n";
        
//         }
        
// };

class ColorSampler: public Sampler{
   public:
        ColorSampler() {}
        uint64_t Scale(uint64_t count) {return count * scale_factor;}
        
        void GetGraph(Graph& g, Graph* subgraph, float delta, Pattern& p) {
            
            printf("sf: %ld\n", scale_factor);
            int c = (int) delta;

            if(delta == 1) {
                return;
            }
            scale_factor = std::pow(delta, p.get_nodes()-1);

            Timer t;
            t.Start();
            auto colors = new int[g.V()];

            std::vector<vidType> new_degrees(g.V(), 0);

            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_int_distribution<int> dist(0, c-1);

            //#pragma omp parallel for schedule(dynamic, 1) shared(colors)
            for (vidType v = 0; v < g.V(); v++) {
                colors[v] = dist(mt);
            }

            t.Stop();
            std::cout << "coloring = " << t.Seconds() << " sec\n";

            t.Start();

            #pragma omp parallel for shared(colors, new_degrees)
            for (vidType src = 0; src < g.V(); src ++) {  
                for(auto dst : g.N(src)) {
                    if(colors[src] == colors[dst]) { //keep edge
                        new_degrees[src] += 1;
                    }
                }
            }


            eidType *new_vertices = custom_alloc_global<eidType>(g.V()+1);
            parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
            auto num_edges = new_vertices[g.V()];

            //auto new_edges = new vidType[num_edges];

            subgraph->allocateFrom(g.V(), num_edges);

            #pragma omp parallel for
            for (vidType src = 0; src < g.V(); src ++) {
                auto begin = new_vertices[src];
                subgraph->fixEndEdge(src, new_vertices[src+1]); // fix row pointers
                eidType offset = 0;
                for (auto dst : g.N(src)) {
                    if (colors[src] == colors[dst]) { // keep edge
                        // new_edges[begin+offset] = dst;
                        subgraph->constructEdge(begin+offset, dst);
                        offset ++;
                    }
                }
            }

        
            t.Stop();
            std::cout << "color induced sparsification fast = " << t.Seconds() << " sec\n";
        
        }
        
};