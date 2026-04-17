// ================================================================
// kernels/boundary.cu
// Condiciones de frontera:
//   Entrada  (ix=0)      : Zou/He velocidad impuesta
//   Salida   (ix=Nx-1)   : extrapolación de segundo orden
//   Paredes  (iy=0,Ny-1) : bounce-back
// ================================================================

#include "../d2q9.cuh"
#include "../headers.cuh"

__global__
void boundary_kernel(float* f,
                      const uint8_t* obstacle,
                      const SimParams p)
{
    // Cada hilo maneja una celda de frontera
    // Usamos un índice lineal y determinamos qué frontera es
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= p.Nx || iy >= p.Ny) return;

    int idx = ix * p.Ny + iy;

    // ── ENTRADA: ix=0, Zou/He velocidad impuesta ─────────────────
    // Conocidos: f0, f2, f3, f4, f6, f7
    // Incógnitas: f1, f5, f8
    // uy = 0 (entrada horizontal uniforme)
    if (ix == 0 && !obstacle[idx]) {
        float ux_in = p.u_inlet_lbm;

        // f conocidos en ix=0
        float f0 = f[idx*Q + 0];
        float f2 = f[idx*Q + 2];
        float f3 = f[idx*Q + 3];
        float f4 = f[idx*Q + 4];
        float f6 = f[idx*Q + 6];
        float f7 = f[idx*Q + 7];

        // Densidad desde Zou/He
        float rho_in = (1.0f / (1.0f - ux_in)) *
                       (f0 + f2 + f4 + 2.0f*(f3 + f6 + f7));

        // f desconocidos
        f[idx*Q + 1] = f3 + (2.0f/3.0f) * rho_in * ux_in;

        f[idx*Q + 5] = f7 - 0.5f*(f2 - f4)
                          + (1.0f/6.0f) * rho_in * ux_in;

        f[idx*Q + 8] = f6 + 0.5f*(f2 - f4)
                          + (1.0f/6.0f) * rho_in * ux_in;
        return;
    }

    // ── SALIDA: ix=Nx-1, extrapolación de segundo orden ──────────
    // f(Nx-1) = 2*f(Nx-2) - f(Nx-3)
    // Solo los f que apuntan hacia afuera del dominio: f3, f6, f7
    if (ix == p.Nx - 1 && !obstacle[idx]) {
        int idx_m1 = (ix-1) * p.Ny + iy;   // ix = Nx-2
        int idx_m2 = (ix-2) * p.Ny + iy;   // ix = Nx-3

        f[idx*Q + 3] = 2.0f*f[idx_m1*Q + 3] - f[idx_m2*Q + 3];
        f[idx*Q + 6] = 2.0f*f[idx_m1*Q + 6] - f[idx_m2*Q + 6];
        f[idx*Q + 7] = 2.0f*f[idx_m1*Q + 7] - f[idx_m2*Q + 7];
        return;
    }

    // ── PARED INFERIOR: iy=0, bounce-back ────────────────────────
    // Direcciones que apuntan hacia +y (entran por iy=0): f2, f5, f6
    if (iy == 0 && !obstacle[idx]) {
        f[idx*Q + 2] = f[idx*Q + OPP[2]];   // f2 ← f4
        f[idx*Q + 5] = f[idx*Q + OPP[5]];   // f5 ← f7
        f[idx*Q + 6] = f[idx*Q + OPP[6]];   // f6 ← f8
        return;
    }

    // ── PARED SUPERIOR: iy=Ny-1, bounce-back ─────────────────────
    // Direcciones que apuntan hacia -y (entran por iy=Ny-1): f4, f7, f8
    if (iy == p.Ny - 1 && !obstacle[idx]) {
        f[idx*Q + 4] = f[idx*Q + OPP[4]];   // f4 ← f2
        f[idx*Q + 7] = f[idx*Q + OPP[7]];   // f7 ← f5
        f[idx*Q + 8] = f[idx*Q + OPP[8]];   // f8 ← f6
        return;
    }
}