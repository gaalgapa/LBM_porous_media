// ================================================================
// kernels/macro.cu
// Calcula variables macroscópicas (rho, ux, uy) desde f.
// Se llama DESPUÉS del streaming, sobre f ya actualizado.
// ================================================================

#include "../d2q9.cuh"
#include "../headers.cuh"

// ── Función de equilibrio ────────────────────────────────────────
__device__ __forceinline__
float feq(int k, float rho, float ux, float uy)
{
    float cu = CX[k]*ux + CY[k]*uy;
    float u2 = ux*ux + uy*uy;
    return W[k] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2);
}

// ── Kernel de inicialización ─────────────────────────────────────
__global__
void initialize_kernel(float* f, float* rho,
                        float* ux, float* uy,
                        const SimParams p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    rho[idx] = p.rho0;
    ux[idx]  = p.u_inlet_lbm;   // velocidad inicial uniforme
    uy[idx]  = 0.0f;

    for (int k = 0; k < Q; k++)
        f[idx*Q + k] = feq(k, p.rho0, p.u_inlet_lbm, 0.0f);
}

// ── Kernel de macroscópicos ──────────────────────────────────────
__global__
void macro_kernel(const float* f,
                   float* rho, float* ux, float* uy,
                   const bool* obstacle,
                   const SimParams p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    // Celdas sólidas: forzar u=0, rho=rho0
    if (obstacle[idx]) {
        rho[idx] = p.rho0;
        ux[idx]  = 0.0f;
        uy[idx]  = 0.0f;
        return;
    }

    float sum_f  = 0.0f;
    float sum_ux = 0.0f;
    float sum_uy = 0.0f;

    for (int k = 0; k < Q; k++) {
        float fk  = f[idx*Q + k];
        sum_f    += fk;
        sum_ux   += fk * CX[k];
        sum_uy   += fk * CY[k];
    }

    rho[idx] = sum_f;
    ux[idx]  = sum_ux / sum_f;
    uy[idx]  = sum_uy / sum_f;
}