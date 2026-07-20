#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 32
__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta){


        int row = blockDim.y*blockIdx.y + threadIdx.y;
        int col = blockDim.x*blockIdx.x + threadIdx.x;

        A += K*blockIdx.y*blockDim.y;
        B += blockIdx.x*blockDim.x;
        C += N*blockIdx.y*blockDim.y + blockIdx.x*blockDim.x;

        // loop A, B tiles 0 thru n in stride of tile
        // compute partial res into C
            
        __shared__ half As[TILE][TILE];
        __shared__ half Bs[TILE][TILE];
        float tmp = 0.0f;
        for (int i=0; i<K; i+=TILE){
            int arow = row;
            int acol = i + threadIdx.x;
            int brow = i + threadIdx.y;
            int bcol = col;
            As[threadIdx.y][threadIdx.x] = (arow<M && acol<K)?A[threadIdx.y*K + threadIdx.x]:__float2half(0.0f); 
            Bs[threadIdx.y][threadIdx.x] = (brow<K && bcol<N)?B[threadIdx.y*N + threadIdx.x]:__float2half(0.0f);
            A += TILE;
            B += TILE*N;
            __syncthreads();
            #pragma unroll
            for (int j=0; j<TILE; j++){
                    tmp += __half2float(As[threadIdx.y][j])* __half2float(Bs[j][threadIdx.x]);
            }
            __syncthreads(); //IMPORTANT: since this is a for loop faster threads might fetch the next tile
        }

        if (col<N && row<M)
            C[threadIdx.y*N + threadIdx.x] = __float2half(alpha* tmp + beta*__half2float(C[threadIdx.y*N + threadIdx.x]));

    }

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 blocksz(32, 32);
    dim3 gridsz((N + blocksz.x - 1)/blocksz.x, (M + blocksz.y -1)/blocksz.y);
    matmul<<<gridsz, blocksz>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
