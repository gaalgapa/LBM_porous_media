// ================================================================
// kernels/collision.cu
// Colisión BGK sin fuerza de cuerpo.
// El flujo es impulsado por la condición de entrada Zou/He,
// no por un gradiente de presión artificial.
// ================================================================

#include "../d2q9.cuh"
#include "../headers.cuh"

__global__
void collision_kernel(const float* f,
                       float* fnew,
                       const float* rho,
                       const float* ux,
                       const float* uy,
                       const uint8_t* obstacle,
                       const SimParams p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    // Celdas sólidas: bounce-back completo
    // fnew[k] = f[opp[k]] — la partícula rebota en dirección opuesta
    if (obstacle[idx]) {
        for (int k = 0; k < Q; k++)
            fnew[idx*Q + k] = f[idx*Q + OPP[k]];
        return;
    }

    float r  = rho[idx];
    float vx = ux[idx];
    float vy = uy[idx];
    float inv_tau = 1.0f / p.tau;

    for (int k = 0; k < Q; k++) {
        float cu   = CX[k]*vx + CY[k]*vy;
        float u2   = vx*vx + vy*vy;
        float fkeq = W[k] * r * (1.0f + 3.0f*cu
                                       + 4.5f*cu*cu
                                       - 1.5f*u2);
        float fk   = f[idx*Q + k];

        // BGK: f_new = f - (1/tau)*(f - feq)
        fnew[idx*Q + k] = fk - inv_tau * (fk - fkeq);
    }
}