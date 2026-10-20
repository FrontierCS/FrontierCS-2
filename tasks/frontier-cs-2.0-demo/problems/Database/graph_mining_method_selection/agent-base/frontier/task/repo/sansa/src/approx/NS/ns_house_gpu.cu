#include "graph.h"
#include "sample.hh"

double sample_house(Graph &g, eidType num_samples);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int64_t num_samples = atoi(argv[2]);

  Timer t;
  auto m = g.init_edgelist(true);
  std::cout << "Number of different edges: " << m << "\n";

  t.Start();
  uint64_t total = 0;
  auto estimate = sample_house(g, num_samples);
  t.Stop();
  std::cout << "Total runtime [data-copy + gpu-compute] = " << t.Seconds() << " sec\n";
  //std::cout << "Estimated count " << FormatWithCommas(estimate) << "\n";
  std::cout << "Estimated count " << estimate << "\n";
}

#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

#define BLK_SZ BLOCK_SIZE
typedef double CounterType;
typedef cub::BlockReduce<CounterType, BLK_SZ> BlockReduce;

template <typename T>
__device__ T random_select_single(T n, curandState &state) {
  return T(ceil(curand_uniform(&state) * n) - 1);
}

__global__ void setup_kernel(curandState *states) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(1234, idx, 0, &states[idx]);
}

__global__ void house_sample(curandState* states, eidType n_samples, float* edges_rng, GraphGPU g, vidType *vlists, vidType max_deg, CounterType *total, AccType *num_hits) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps

  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)*2];
  vidType *vlist1 = &vlists[int64_t(warp_id)*int64_t(max_deg)*2+int64_t(max_deg)];
  CounterType counter = 0;
  __shared__ CounterType scale[WARPS_PER_BLOCK];
  __shared__ vidType idx0[WARPS_PER_BLOCK];
  __shared__ vidType idx1[WARPS_PER_BLOCK];
  //__shared__ vidType idx2[WARPS_PER_BLOCK];
  __shared__ vidType vs02[WARPS_PER_BLOCK][2];
  __shared__ vidType vs12[WARPS_PER_BLOCK][2];
  eidType m = g.E() / 2;
  //if (thread_lane == 0) idx1[warp_lane] = 0; __syncwarp();

  // each warp draws one sample
  for (eidType sample_id = warp_id; sample_id < n_samples; sample_id += num_warps) {
    // sample an edge
    eidType eid = ceil(edges_rng[sample_id]*m) - 1;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    if (v0_size < 3) continue;
    vidType v1_size = g.getOutDegree(v1);
    if (v1_size < 3) continue;

    // intersection of v0 and v1's neighbors
    auto c0 = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    __syncwarp();
    if (c0 == 0) continue;

    // sample a common neighbor of v0 and v1's as v2
    if (thread_lane == 0) idx0[warp_lane] = ceil(curand_uniform(&states[warp_id]) * c0) - 1;
    __syncwarp();
    vidType v2 = vlist[idx0[warp_lane]];
 
    if (thread_lane == 0) scale[warp_lane] = CounterType(c0); __syncwarp();

    auto u = v1 > v2 ? v2 : v1;
    auto w = v1 > v2 ? v1 : v2;
    if (thread_lane == 0) {
      vs12[warp_lane][0] = u;
      vs12[warp_lane][1] = w;
    }
    __syncwarp();

    auto c1 = v0_size - 2;
    //auto c1 = difference_set(g.N(v0), v0_size, vs12[warp_lane], 2, vlist);
    if (thread_lane == 0) idx1[warp_lane] = ceil(curand_uniform(&states[warp_id]) * c1) - 1;
    __syncwarp();
    auto v3 = difference_sample(g.N(v0), v0_size, vs12[warp_lane], 2, idx1[warp_lane]);
    //vidType v3 = vlist[idx1[warp_lane]];
    vidType v3_degree = g.getOutDegree(v3);
 
    if (thread_lane == 0) scale[warp_lane] *= c1; __syncwarp();

    u = v0 > v2 ? v2 : v0;
    w = v0 > v2 ? v0 : v2;
    if (thread_lane == 0) {
      vs02[warp_lane][0] = u;
      vs02[warp_lane][1] = w;
    }
    __syncwarp();
    //auto cx = difference_set(g.N(v1), v1_size, vs02[warp_lane], 2, vlist1);
    auto cx = difference_set(g.N(v1), v1_size, vs02[warp_lane], 2, vlist1);
    __syncwarp();
    //if (cx != v1_size - 2 && thread_lane == 0)
    //  printf("sample_id=%ld, wid=%d, eid=%ld, v0=%d (deg=%d), v1=%d (deg=%d), idx0=%d, v2=%d, idx1=%d, v3=%d, c0=%d, c1=%d, cx=%d\n", 
    //                              sample_id, warp_id, eid, v0, v0_size, v1, v1_size, idx0[warp_lane], v2, idx1[warp_lane], v3, c0, c1, cx);
    //assert(cx == v1_size - 2);
    auto num = intersect_num(g.N(v3), v3_degree, vlist1, cx);
    auto c2 = warp_reduce<vidType>(num);
    //if (c2 < 1) continue;
    //__syncwarp();

    if (thread_lane == 0) atomicAdd(num_hits, 1);

    if (thread_lane == 0) scale[warp_lane] *= c2; __syncwarp();
    if (thread_lane == 0) counter += scale[warp_lane];
    //auto scale = CounterType(c0) * CounterType(c1) * CounterType(c2);
    //if (thread_lane == 0) counter += scale;
    //if (thread_lane == 0)
    //  printf("sample_id=%ld, wid=%d, eid=%ld, v0=%d (deg=%d), v1=%d (deg=%d), idx1=%d, v2=%d, idx3=%d, v3=%d, c0=%d, c1=%d, c2=%d, cx=%d, scale=%f\n", 
    //                              sample_id, warp_id, eid, v0, v0_size, v1, v1_size, idx1[warp_lane], v2, idx3, v3, c0, c1, c2, cx, scale);
    __syncwarp();
  }
  auto block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

#define PAR_FACTOR 10
double sample_house(Graph &g, eidType num_samples) {
  vidType nv = g.num_vertices();
  eidType ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t memsize = print_device_info(0);
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

  Timer t;
  t.Start();
  GraphGPU gg(g);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "time[copy graph to gpu] = " << t.Seconds() << " sec\n";

  t.Start();
  auto m = g.E()/2;
  gg.copy_edgelist_to_device(m, g);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "copy edgelist to gpu: " << m << " edges\n";
  std::cout << "time[copy edgelist to gpu] = " << t.Seconds() << " sec\n";

  size_t nwarps = WARPS_PER_BLOCK;
  size_t nthreads = BLK_SZ;
  size_t nblocks = (num_samples-1)/WARPS_PER_BLOCK+1;
  int max_blocks_per_SM = 0;
  max_blocks_per_SM = maximum_residency(house_sample, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t max_resident_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  std::cout << "max_resident_blocks = " << max_resident_blocks << "\n";
  if (nblocks > max_resident_blocks*PAR_FACTOR) nblocks = max_resident_blocks*PAR_FACTOR;
  if (nblocks > 65536) nblocks = 65536;

  size_t per_block_vlist_size = nwarps * size_t(md) * sizeof(vidType) * 2;
  size_t nb = (memsize - mem_graph) / per_block_vlist_size;
  //if (nb < nblocks) nblocks = nb;
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size << " Bytes, " << list_size/(1024*1024) << " MB\n";
  vidType *frontier_list; // each warp has a vertex set; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  CounterType h_counter = 0, *d_counter;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_counter, sizeof(CounterType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_counter, &h_counter, sizeof(CounterType), cudaMemcpyHostToDevice));
  AccType h_num_hits = 0, *d_num_hits;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_hits, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_num_hits, &h_num_hits, sizeof(AccType), cudaMemcpyHostToDevice));

  // generate random numbers for edge sampling
  t.Start();
  float* d_rands;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_rands, num_samples * sizeof(float)));
  curandGenerator_t curand_gen; // used to generate random numbers on GPU
  auto status = curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
  if (status != CURAND_STATUS_SUCCESS) {
    printf("Error encountered in generating handle\n");  
  }
  // set seed
  status = curandSetPseudoRandomGeneratorSeed(curand_gen ,1234ULL);  
  if (status != CURAND_STATUS_SUCCESS) {
    printf("Error encountered in setting seed\n");
  }
  // Generate N floats on device between 0.0 and 1.0
  CURAND_CHECK(curandGenerateUniform(curand_gen, d_rands, num_samples));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "RNG per sample runtime [gpu] = " << t.Seconds() << " sec\n";

  // allocate a RNG for each warp
  curandState *d_states;
  cudaMalloc(&d_states, sizeof(curandState)*nblocks*nwarps);
  setup_kernel<<<nblocks, nwarps>>>(d_states);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  std::cout << "CUDA sampling house (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  t.Start();
  std::cout << "Launching kernel: num_samples=" << num_samples << "\n";
  house_sample<<<nblocks, nthreads>>>(d_states, num_samples, d_rands, gg, frontier_list, md, d_counter, d_num_hits);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "runtime [gpu] = " << t.Seconds() << " sec\n";

  CUDA_SAFE_CALL(cudaMemcpy(&h_counter, d_counter, sizeof(CounterType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_num_hits, d_num_hits, sizeof(AccType), cudaMemcpyDeviceToHost));

  std::cout << "counter = " << h_counter << " avg_count_per_sample = " << double(h_counter)/double(num_samples) << "\n";
  std::cout << "num_hits = " << h_num_hits << " hit_rate = " << double(h_num_hits)/double(num_samples) << "\n";

  double estimate = (h_counter * double(m)) / double(num_samples);

  // clean up
  CUDA_SAFE_CALL(cudaFree(d_counter));
  CURAND_CHECK(curandDestroyGenerator(curand_gen));
  CUDA_SAFE_CALL(cudaFree(d_rands));
  return estimate;
}

