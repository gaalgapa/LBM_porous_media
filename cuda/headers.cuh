// ================================================================
// headers.cuh
// Declaraciones de todos los kernels y funciones del programa.
// ================================================================

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

// ── Estructura de parámetros ─────────────────────────────────────
// Se llena al inicio desde config.json y se pasa a todos los
// kernels que lo necesiten. Vive en memoria de host y device.
struct SimParams {
    // Grilla
    int   Nx, Ny;
    // LBM
    float tau;
    float nu_lbm;
    float u_inlet_lbm;   // velocidad de entrada Zou/He
    float rho0;          // densidad inicial
    // Convergencia
    int   t_max;
    float max_diff;
    int   save_every;    // cada cuántos pasos guardar VTI
    int   ckpt_every;    // cada cuántos pasos guardar checkpoint
    // Conversión a unidades físicas
    float c_l;           // [m/celda]
    float c_t;           // [s/paso]
    float rho_phy;       // [kg/m³]
    float mu_phy;        // [Pa·s]
    // Identificación
    int   seed;
    char  run_dir[512];  // ruta donde guardar resultados
};

// ── Kernels de simulación ────────────────────────────────────────

// Inicialización: f = feq(rho0, 0, 0)
__global__ void initialize_kernel(float* f,
                                   float* rho,
                                   float* ux,
                                   float* uy,
                                   const SimParams p);

// Cálculo de variables macroscópicas desde f
__global__ void macro_kernel(const float* f,
                              float* rho,
                              float* ux,
                              float* uy,
                              const uint8_t* obstacle,
                              const SimParams p);

// Colisión BGK
__global__ void collision_kernel(const float* f,
                                  float* fnew,
                                  const float* rho,
                                  const float* ux,
                                  const float* uy,
                                  const uint8_t* obstacle,
                                  const SimParams p);

// Condiciones de frontera
__global__ void boundary_kernel(float* fnew,
                                 const uint8_t* obstacle,
                                 const SimParams p);

// Streaming
__global__ void streaming_kernel(const float* fnew,
                                  float* f,
                                  const uint8_t* obstacle,
                                  const SimParams p);

// Error de convergencia
__global__ void error_kernel(const float* ux,
                              const float* uy,
                              const float* ux_prev,
                              const float* uy_prev,
                              float* num,
                              float* den,
                              const uint8_t* obstacle,
                              const SimParams p);

// ── Funciones de entrada/salida ──────────────────────────────────

void write_vti(const std::string& fpath,
               const float* ux,
               const float* uy,
               const float* rho,
               const uint8_t*  obstacle,
               const SimParams& p,
               int timestep);

void write_metrics_csv(const std::string& fpath,
                       const float* rho,
                       const float* ux,
                       const SimParams& p,
                       int seed,
                       int t_final,
                       bool converged);

void write_profile_csv(const std::string& fpath,
                       const float* ux,
                       const float* uy,
                       const uint8_t*  obstacle,
                       const SimParams& p,
                       int seed);

void save_checkpoint(const std::string& fpath,
                     const float* f,
                     const SimParams& p,
                     int timestep);

bool load_checkpoint(const std::string& fpath,
                     float* f,
                     const SimParams& p,
                     int& timestep);

SimParams load_config(const std::string& config_path,
                      const std::string& run_dir,
                      int seed);