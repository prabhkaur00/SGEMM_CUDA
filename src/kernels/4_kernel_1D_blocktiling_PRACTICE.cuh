#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.


  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM*BK];
  __shared__ float Bs[BK*BN];

  // Move blocktile to beginning of A's row and B's column
  A+=cRow*K*BM;
  B+=cCol*BN;
  C+=crow*K*BM + cCol*BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  int innerColA = threadIdx.x%BK;
  int innerRowA = threadIdx.x/BK;
  int innerColB = threadIdx.x%BN;
  int innerRowB = threadIdx.x/BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};                             
  // outer loop over block tiles
  for (int i=0; i+=BK; i<K) {
    // populate the SMEM caches
    As[innerRowA*BK + innerColA] = A[innerRowA*K * innerColA];
    Bs[innerRowB*BN + innerColB] = B[innerRowB*K * innerColB];
    __syncthreads();
    // advance blocktile
    A+=BK;
    B+=BK*N;
    // calculate per-thread results
    for (int dotIdx=0; dotIdx<BK; dotIdx++) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      for (int i=0; i<TM; i++){
        threadResults[i]+=As[i*BK + dotIdx]*Bs[dotIdx*BN];
      }
    }
    __syncthreads();
  }

  // write out the results
  for (int i=0; i<TM; i++){
    C[]
  }

}