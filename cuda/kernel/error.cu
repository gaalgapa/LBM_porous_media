// ================================================================
// kernels/error.cu
// Calcula el error relativo RMS de velocidad para convergencia:
//   error = sqrt( Σ|u_new - u_prev|² / Σ|u_new|² )
// Solo sobre celdas de fluido.
// ================================================================

#include "../d2q9.cuh"
#include "../headers.cuh"

__global__
void error_kernel(const float* ux,
                   const float* uy,
                   const float* ux_prev,
                   const float* uy_prev,
                   float* num,
                   float* den,
                   const bool* obstacle,
                   const SimParams p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    // Celdas sólidas no contribuyen al error
    if (obstacle[idx]) {
        num[idx] = 0.0f;
        den[idx] = 0.0f;
        return;
    }

    float dux = ux[idx] - ux_prev[idx];
    float duy = uy[idx] - uy_prev[idx];

    num[idx] = dux*dux + duy*duy;
    den[idx] = ux[idx]*ux[idx] + uy[idx]*uy[idx];
}