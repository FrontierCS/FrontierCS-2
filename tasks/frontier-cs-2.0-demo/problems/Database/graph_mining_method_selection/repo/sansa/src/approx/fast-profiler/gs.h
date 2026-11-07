uint64_t ExactCount(Graph& g, Pattern& p, Sampler& s, int dag, int chunk_size) {
  
  uint64_t counter = 0;

  if (p.is_house()) {
    #include "house.h"
  } else if (p.is_pentagon()) {
    #include "pentagon.h"
  } else if (p.is_rectangle()) {
    #include "rectangle.h"
  } else if (p.is_diamond()) {
    #include "diamond.h"
  } else if(p.is_triangle()) {
    #include "triangle.h"
  } else if(p.is_9clique()) {
    #include "9clique.h"
  } else if(p.is_5clique()) {
    #include "5clique.h"
  } else if(p.is_4clique()) {
    #include "4clique.h"
  } else if(p.is_5path()) {
    #include "5path.h"
  } else if (p.is_6clique()) {
    #include "6clique.h"
  }

  printf("p_hat = %ld * %lu = %lu \n", s.scale_factor, counter, s.Scale(counter));

  return s.Scale(counter);
}