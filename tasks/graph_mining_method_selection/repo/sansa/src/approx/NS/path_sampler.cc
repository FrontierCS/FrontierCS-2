#include "graph.h"
#include "sample.hh"

// sample a size-k connected subgraph in graph
double connected_subgraph_sampler(Graph &g, int k, eidType eid, default_random_engine & rand_generator, std::vector<vidType> &vlist) {
  double prob_inverse = 1.0;
  auto v0 = g.get_src(eid);
  auto v1 = g.get_dst(eid);
  std::set<vidType> vs;
  vs.insert(v0);
  vs.insert(v1);
  std::set<vidType> open_vs;
  open_vs.emplace(v0);
  open_vs.emplace(v1);
  vlist.push_back(v0);
  vlist.push_back(v1);

  uint32_t total_d_extending_edge = 0;
  for (int j = 0; j < k - 2; j++) {
    total_d_extending_edge = 0;
    vector<vidType> open_set_neighbor;
    for (auto u : open_vs) {
      auto begin_idx = g.edge_begin(u);
      auto end_idx = g.edge_end(u);
      for (auto idx = begin_idx; idx < end_idx; idx++) {
        auto w = g.getEdgeDst(idx);
        if (vs.find(w) != vs.end()) continue;
        total_d_extending_edge++;
        open_set_neighbor.push_back(idx);
      }
    }
    if (total_d_extending_edge == 0) return 0;
    prob_inverse *= total_d_extending_edge;
    uniform_int_distribution <uint32_t> next_edge_distribution(0, total_d_extending_edge - 1);
    auto edge_idx = open_set_neighbor[next_edge_distribution(rand_generator)];
    auto u0 = g.get_src(edge_idx);
    auto u1 = g.get_dst(edge_idx);
    //vs.insert(u0);
    vs.insert(u1);
    open_vs.emplace(u1);
    vlist.push_back(u1);
  }
  assert(vlist.size() == k);
  return prob_inverse;
}

// sample a size-k path in graph
double path_sampler(Graph &g, int k, eidType eid, default_random_engine & rand_generator, std::vector<vidType> &vlist) {
  double prob_inverse = 1.0;
  auto v0 = g.get_src(eid);
  auto v1 = g.get_dst(eid);
  std::set<vidType> vs;
  vs.insert(v0);
  vs.insert(v1);
  std::set<vidType> open_vs;
  open_vs.emplace(v0);
  open_vs.emplace(v1);
  vlist.push_back(v0);
  vlist.push_back(v1);

  uint32_t total_d_extending_edge = 0;
  for (int j = 0; j < k - 2; j++) {
    total_d_extending_edge = 0;
    vector<vidType> open_set_neighbor;
    for (auto u : open_vs) {
      auto begin_idx = g.edge_begin(u);
      auto end_idx = g.edge_end(u);
      for (auto idx = begin_idx; idx < end_idx; idx++) {
        auto w = g.getEdgeDst(idx);
        if (vs.find(w) != vs.end()) continue;
        total_d_extending_edge++;
        open_set_neighbor.push_back(idx);
      }
    }
    if (total_d_extending_edge == 0) return 0;
    prob_inverse *= total_d_extending_edge;
    uniform_int_distribution <uint32_t> next_edge_distribution(0, total_d_extending_edge - 1);
    auto edge_idx = open_set_neighbor[next_edge_distribution(rand_generator)];
    auto u0 = g.get_src(edge_idx);
    auto u1 = g.get_dst(edge_idx);
    vs.insert(u0);
    vs.insert(u1);
    open_vs.erase(u0);
    open_vs.emplace(u1);
    vlist.push_back(u1);
  }
  assert(vlist.size() == k);
  return prob_inverse;
}

/*
double path_sampler(Graph &g, int k, eidType eid, default_random_engine & gen, std::set<vidType> &vertex_set) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);

    VertexSet vs0, vs1;
    vs0.add(v0);
    vs1.add(v1);
    vertex_set.add(v0);
    vertex_set.add(v1);

    auto y0 = difference_set(g.N(v0), vs1);
    auto y1 = difference_set(g.N(v1), vs0);
    vidType c0 = y0.size() + y1.size();
    if (c0 == 0) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    vidType v2, vj;
    if (idx0 < y0.size()) v2 = y0[idx0];
    else v2 = y1[idx0];
    vertex_set.add(v2);
    auto vj = v2;

    double scale = 1.;
    for (int j = 2; j < k; j++) {
      VertexSet vs;
      vs.add(vj);
      auto yj = difference_set(g.N(vj), vs);
      if (c == 0) { scale = 0; break; }
      auto id = random_select_single<vidType>(0, c-1, gen);
      vj = [id];
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
  }
}
*/
