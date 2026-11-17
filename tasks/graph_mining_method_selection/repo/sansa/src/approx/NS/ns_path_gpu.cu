#include "graph.h"
#include "sample.hh"

double sample_path(Graph &g, int k, eidType num_samples);

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <path_nodes> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();

  int k = atoi(argv[2]); // number of vertices in the path
  assert(k > 2);
  std::cout << "Finding " << k << "-path\n";
  eidType num_samples = atoi(argv[3]);
  std::cout << "num_samples: " << num_samples << "\n";

  Timer t;
  g.init_simple_edgelist();
  t.Start();
  double estimate = sample_path(g, k, num_samples);
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
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
#define MAX_PATTERN_SIZE 9
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

__global__ void k_path_sample(int k, curandState* states, eidType n_samples, float* edges_rng, GraphGPU g, vidType *vlists, vidType max_deg, CounterType *total, AccType *num_hits) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  CounterType counter = 0;
  //vidType *vlist0 = &vlists[int64_t(warp_id)*int64_t(max_deg)];

  __shared__ vidType list_size[WARPS_PER_BLOCK];
  __shared__ vidType idx[WARPS_PER_BLOCK];
  __shared__ CounterType scale[WARPS_PER_BLOCK];
  __shared__ vidType vs[WARPS_PER_BLOCK][MAX_PATTERN_SIZE-1];
  __shared__ vidType vs_size[WARPS_PER_BLOCK];

  // each warp draws one sample
  for (eidType sample_id = warp_id; sample_id < n_samples; sample_id += num_warps) {
    if (thread_lane == 0) scale[warp_lane] = 1;
    __syncwarp();

    // sample an edge
    eidType eid = ceil(edges_rng[sample_id]*g.E()) - 1;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);

    // buffer existing nodes
    if (thread_lane == 0) {
      vs[warp_lane][0] = v0 < v1 ? v0 : v1;
      vs[warp_lane][1] = v0 < v1 ? v1 : v0;
      vs_size[warp_lane] = 2;
    }
    __syncwarp();

    vidType v = v1;
    vidType v_deg = g.getOutDegree(v);

    for (int j = 2; j < k; j++) {
      vidType c = 0;
      if (j == k - 1) {
        auto num = difference_num(g.N(v), v_deg, vs[warp_lane], vs_size[warp_lane], v0);
        c = warp_reduce<vidType>(num);
      } else {
        //c = difference_set(g.N(v), v_deg, vs[warp_lane], vs_size[warp_lane], vlist0);
        auto num = difference_num(g.N(v), v_deg, vs[warp_lane], vs_size[warp_lane]);
        c = warp_reduce<vidType>(num);
        if (c < 1) {
          if (thread_lane == 0) scale[warp_lane] = 0; __syncwarp();
          break;
        }
        if (thread_lane == 0) idx[warp_lane] = ceil(curand_uniform(&states[warp_id]) * c) - 1;
        __syncwarp();
        //v = vlist0[idx[warp_lane]];
        v = difference_sample(g.N(v), v_deg, vs[warp_lane], vs_size[warp_lane], idx[warp_lane]);
        v_deg = g.getOutDegree(v);
        if (thread_lane == 0) {
          int i = 0;
          for (; i < j; i++) {
            if (v < vs[warp_lane][i]) {
              for (int ii = j-1; ii >= i; ii --) {
                vs[warp_lane][ii+1] = vs[warp_lane][ii];
              }
              break;
            }
          }
          vs[warp_lane][i] = v;
          vs_size[warp_lane] ++;
        }
        //warp_sort(vs[warp_lane], vs_size[warp_lane]);
        __syncwarp();
      }
      if (c < 1) {
        if (thread_lane == 0) scale[warp_lane] = 0;
        __syncwarp();
        break;
      }
      if (thread_lane == 0) scale[warp_lane] *= c;
      __syncwarp();
    }
    if (thread_lane == 0) atomicAdd(num_hits, 1);
    if (thread_lane == 0) counter += scale[warp_lane];
  }
  auto block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

#define PAR_FACTOR 32
double sample_path(Graph &g, int k, eidType num_samples) {
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
  gg.copy_edgelist_to_device(g.E(), g);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "time[copy edgelist to gpu] = " << t.Seconds() << " sec\n";

  size_t nwarps = WARPS_PER_BLOCK;
  size_t nthreads = BLK_SZ;
  size_t nblocks = (num_samples-1)/WARPS_PER_BLOCK+1;
  int max_blocks_per_SM = 0;
  max_blocks_per_SM = maximum_residency(k_path_sample, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t max_resident_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  std::cout << "max_resident_blocks = " << max_resident_blocks << "\n";
  if (nblocks > max_resident_blocks*PAR_FACTOR) nblocks = max_resident_blocks*PAR_FACTOR;
  if (nblocks > 65536) nblocks = 65536;

  std::cout << "CUDA sampling " << k << "-path (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  vidType *frontier_list; // each warp has a vertex set; each set has size of max_degree
/*
  size_t per_block_vlist_size = nwarps * size_t(md) * sizeof(vidType);
  std::cout << "per-block frontier list size: " << per_block_vlist_size/(1024*1024) << " MB\n";
  //size_t nb = (memsize - mem_graph) / per_block_vlist_size;
  //if (nb < nblocks) nblocks = nb;
  size_t list_size = nblocks * per_block_vlist_size;
  std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
*/
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
  auto status = curandCreateGenerator (&curand_gen ,CURAND_RNG_PSEUDO_DEFAULT );  
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

  t.Start();
  std::cout << "Launching kernel: num_samples=" << num_samples << "\n";
  k_path_sample<<<nblocks, nthreads>>>(k, d_states, num_samples, d_rands, gg, frontier_list, md, d_counter, d_num_hits);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  std::cout << "runtime [gpu] = " << t.Seconds() << " sec\n";

  CUDA_SAFE_CALL(cudaMemcpy(&h_counter, d_counter, sizeof(CounterType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&h_num_hits, d_num_hits, sizeof(AccType), cudaMemcpyDeviceToHost));

  std::cout << "counter = " << h_counter << " avg_count_per_sample = " << double(h_counter)/double(num_samples) << "\n";
  std::cout << "num_hits = " << h_num_hits << " hit_rate = " << double(h_num_hits)/double(num_samples) << "\n";

  double estimate = h_counter * CounterType(g.E()) / CounterType(num_samples);

  // clean up
  CUDA_SAFE_CALL(cudaFree(d_counter));
  CURAND_CHECK(curandDestroyGenerator(curand_gen));
  CUDA_SAFE_CALL(cudaFree(d_rands));
  return estimate;
}

