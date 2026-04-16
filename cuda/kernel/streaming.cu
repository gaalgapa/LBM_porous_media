// ================================================================
// kernels/streaming.cu
// Advección: propaga fnew hacia los vecinos correctos.
// Periodicidad en X desactivada — las fronteras las maneja
// boundary_kernel. Solo streaming interno.
// ================================================================

#include "../d2q9.cuh"
#include "../headers.cuh"

__global__
void streaming_kernel(const float* fnew,
                       float* f,
                       const bool* obstacle,
                       const SimParams p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    for (int k = 0; k < Q; k++) {
        int ix_next = ix + CX[k];
        int iy_next = iy + CY[k];

        // No hacer streaming fuera del dominio
        if (ix_next < 0 || ix_next >= p.Nx) continue;
        if (iy_next < 0 || iy_next >= p.Ny) continue;

        int idx_next = ix_next * p.Ny + iy_next;
        f[idx_next*Q + k] = fnew[idx*Q + k];
    }
}