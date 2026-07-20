#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 32
#define BK 32
#define BN 32
#define TM 2
#define TK 4
#define TN 2
__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta){
        // blockDim.y = BM/TM, blockDim.x = BN/TN

        int cBlockStartX = blockDim.y*blockIdx.y*TM;
        int cBlockStartY = blockDim.x*blockIdx.x*TN;

        int aBlockStartX = blockDim.y*blockIdx.y*TM;
        int aBlockStartY = blockDim.x*blockIdx.x*TK;

        int bBlockStartX = blockDim.y*blockIdx.y*TK;
        int bBlockStartY = blockDim.x*blockIdx.x*TN;

        int cTileStX = cBlockStartX + threadIdx.y*TM; // my CX indices [tileStX, tileStX + TM - 1]
        int cTileStY = cBlockStartY + threadIdx.x*TN; // my CY indices [tileStY, tileStY + TN - 1]

        // loop A, B tiles 0 thru n in stride of tile
        // compute partial res into C
            
        __shared__ half As[BM][BK];
        __shared__ half Bs[BK][BN];

        float tmp[TM][TN] = {0.0f};
        half areg[TM] = {0.0f};
        half breg[TN] = {0.0f};
        
        int numThreads = blockDim.x*blockDim.y;
        int absind = threadIdx.x + threadIdx.y*blockDim.x;
        int asx = absind/BK;
        int asy = absind%BK;

        int bsx = absind/BN;
        int bsy = absind%BN;

        int atilex = cTileStX;
        int atiley = aBlockStartY + threadIdx.x*TK;
        int btilex = bBlockStartX + threadIdx.y*TK;
        int btiley = cTileStY;
        
        // loop over all C blocks 
        for (int i=0; i<K; i+=BK){
            // copy TM*TK for As, TK*TN for Bs
            for (int idx = absind; idx < BM*BK; idx += numThreads) {
                asx = idx / (BK/4);
                asy = idx % (BK/4);
                int gRow = cBlockStartX + asx;   
                int gCol = i + asy;  
                float* copyadd = &A[gRow*K + gCol];
                float4 * vectorizedaddress = reinterpret_cast<float4*>(copyadd);         
                *retinerpret_cast<float4*>(&As[asx][asy]) = *vectorizedaddress; // *(A + val), based on ptr type those many bytes are fetched
                // LHS casted to enable STS.128
            }
            for (int idx = absind; idx < BK*BN; idx += numThreads) {
                bsx = idx / BN;
                bsy = idx % BN;
                int gRow = i + bsx;              
                int gCol = cBlockStartY + bsy;  
                Bs[bsx][bsy] = (gRow < K && gCol < N) ? B[gRow*N + gCol] : __float2half(0.0f);
            }
            __syncthreads();

            for (int idx = 0 ; idx<BK; idx++){
                int asrow = threadIdx.y*TM;
                int acsol = threadIdx.x*TK;
                int bsrow = threadIdx.y*TK;
                int bscol = threadIdx.x*TN;
                int el = 0;
                for (int aa=asrow; aa<asrow+TM; aa+=1){
                    areg[el] = As[aa][idx];
                    el++;
                }
                el = 0;
                // idx is brow
                for (int bb=bscol; bb<bscol+TN ; bb+=1){
                    breg[el] = Bs[idx][bb];
                    el++;
                }
                #pragma unroll
                for (int ix = 0; ix<TM; ix++){
                    for (int iy = 0; iy<TN; iy++){
                        tmp[ix][iy] += __half2float(areg[ix])*__half2float(breg[iy]);
                    }
                }
            }
            __syncthreads(); //IMPORTANT: since this is a for loop faster threads might fetch the next tile
        }

        //write to C
        // int cBlockStartX = blockDim.y*blockIdx.y*TM;
        // int cBlockStartY = blockDim.x*blockIdx.x*TN;

        for (int ix = 0; ix< TM; ix++){
            for (int iy = 0; iy<TN; iy++){
                int cx = (ix + cTileStX);
                int cy = cTileStY + iy;
                if (cx<M && cy<N){
                    C[cx*N + cy] = __float2half(alpha*tmp[ix][iy] + beta*__half2float(C[cx*N + cy]));
                }
            }
        }


    }

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 blocksz((BN + TN - 1)/TN, (BM + TM - 1)/TM);
    dim3 gridsz((N + BN - 1)/BN, (M  + BM - 1)/BM);
    matmul<<<gridsz, blocksz>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
