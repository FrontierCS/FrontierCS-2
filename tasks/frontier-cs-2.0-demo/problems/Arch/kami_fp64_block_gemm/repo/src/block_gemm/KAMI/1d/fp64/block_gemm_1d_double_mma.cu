#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

using namespace nvcuda;

#ifndef M_BLOCK
#define M_BLOCK 64
#define N_BLOCK 64
#define K_BLOCK 64
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 4
#endif

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 8

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

#define NUM_PIPE_N_BLOCK 1
#define NUM_PIPE_K_BLOCK 1

const int shmem_size_block_gemm = (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) * (K_BLOCK / NUM_PIPE_K_BLOCK) * sizeof(double);

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (NUM_RANK_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__global__ void block_gemm_1d_double_mma(const double *A, const double *B, double *C, const double alpha, const double beta)
{
}

int main(int argc, char *argv[])
{
    int device_id = 0;
    cudaDeviceProp prop;
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "GPU " << prop.pciDeviceID << " Model: " << prop.name << std::endl;

    double *h_A = (double *)malloc(M_BLOCK * K_BLOCK * sizeof(double));
    double *h_B = (double *)malloc(K_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK * K_BLOCK; i++)
    {
        h_A[i] = 1;

        h_A[i] = i;
    }

    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {
        h_B[i] = 1;

        h_B[i] = i;
    }

    double *d_A, *d_B;
    double *d_C;

    cudaMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(double));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(double));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(double));

    cudaMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(double));

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(block_gemm_1d_double_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    block_gemm_1d_double_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);

    cudaEventRecord(start);
    block_gemm_1d_double_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    long long numOpsPerMatrix = 2LL * M_BLOCK * N_BLOCK * K_BLOCK;
    long long totalOps = numOpsPerMatrix * BLOCKS_PER_GRID * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << tflops << "," << NUM_ALLOC_RANK_BLOCK * 32 << std::endl;

    double *h_C = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(double), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < N_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] += h_A[i * K_BLOCK + k] * h_B[j * K_BLOCK + k];
            }
        }
    }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs(h_C[i] / NUM_ITER - h_C_ref[i]) > 1e-6)
        {
            std::cout << "Error: " << i << " " << h_C[i] << " " << h_C_ref[i] << std::endl;
            error = 1;
            break;
        }
    }

    if (!error)
    {
        std::cout << "Validation successful!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
