// ================================================================
// io/write_csv.cu
// Escribe métricas finales y perfiles de velocidad en CSV.
// Estos son los archivos que lee postprocess.py.
// ================================================================

#include <cstdint>
#include <iostream>       // Para solucionar los errores de std::cerr y std::cout
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h> // Buena práctica de seguridad
#include "../d2q9.cuh"    // Para solucionar el error de "Q is undefined"
#include "../headers.cuh"

// ── Métricas finales ─────────────────────────────────────────────
void write_metrics_csv(const std::string& fpath,
                        const float* d_rho,
                        const float* d_ux,
                        const SimParams& p,
                        int seed,
                        int t_final,
                        bool converged)
{
    int N = p.Nx * p.Ny;

    // Copiar a host
    std::vector<float> h_rho(N), h_ux(N);
    cudaMemcpy(h_rho.data(), d_rho,
               N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux.data(),  d_ux,
               N*sizeof(float), cudaMemcpyDeviceToHost);

    // ── Velocidad de Darcy ───────────────────────────────────────
    // Promedio espacial de ux en todo el dominio en lattice units
    double sum_ux = 0.0;
    int    n_fluid = 0;
    for (int i = 0; i < N; i++) {
        sum_ux += h_ux[i];
        n_fluid++;
    }
    float U_darcy = (n_fluid > 0)
                    ? (float)(sum_ux / n_fluid)
                    : 0.0f;

    // ── Presión entrada y salida ─────────────────────────────────
    // Promedio de rho en ix=0 y ix=Nx-1
    double rho_in  = 0.0, rho_out = 0.0;
    for (int iy = 0; iy < p.Ny; iy++) {
        rho_in  += h_rho[0          * p.Ny + iy];
        rho_out += h_rho[(p.Nx - 1) * p.Ny + iy];
    }
    rho_in  /= p.Ny;
    rho_out /= p.Ny;

    // dP/L en lattice units (se convierte a físico en postprocess.py)
    float dP_L = (float)((rho_in - rho_out) / 3.0 / p.Nx);

    // ── Escribir CSV ─────────────────────────────────────────────
    // Si el archivo no existe, escribir header
    std::ifstream check(fpath);
    bool write_header = !check.good();
    check.close();

    std::ofstream f(fpath, std::ios::app);

    if (write_header)
        f << "seed,t_final,converged,U_darcy,rho_inlet,"
          << "rho_outlet,dP_L\n";

    f << seed        << ","
      << t_final     << ","
      << (converged ? 1 : 0) << ","
      << U_darcy     << ","
      << rho_in      << ","
      << rho_out     << ","
      << dP_L        << "\n";

    f.close();
}

// ── Perfil de velocidad promediado en x ─────────────────────────
void write_profile_csv(const std::string& fpath,
                        const float* d_ux,
                        const float* d_uy,
                        const bool*  d_obstacle,
                        const SimParams& p,
                        int seed)
{
    int N = p.Nx * p.Ny;

    // Copiar a host
    std::vector<float>   h_ux(N), h_uy(N);
    std::vector<uint8_t>    h_obs(N);

    cudaMemcpy(h_ux.data(),      d_ux,
               N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(),      d_uy,
               N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obs.data(), d_obstacle,
                N*sizeof(uint8_t),  cudaMemcpyDeviceToHost);

    // Para cada y: promediar ux y uy sobre todas las x de fluido
    std::vector<double> sum_ux(p.Ny, 0.0);
    std::vector<double> sum_uy(p.Ny, 0.0);
    std::vector<double> obs_frac(p.Ny, 0.0);
    std::vector<int>    count(p.Ny, 0);

    for (int ix = 0; ix < p.Nx; ix++) {
        for (int iy = 0; iy < p.Ny; iy++) {
            int idx = ix * p.Ny + iy;
            if (!h_obs[idx]) {
                sum_ux[iy]   += h_ux[idx];
                sum_uy[iy]   += h_uy[idx];
                count[iy]++;
            }
            obs_frac[iy] += h_obs[idx] ? 1.0 : 0.0;
        }
    }

    // Escribir CSV
    std::ofstream f(fpath);
    f << "y,ux_mean,uy_mean,obstacle_fraction\n";

    for (int iy = 0; iy < p.Ny; iy++) {
        float ux_m = (count[iy] > 0)
                     ? (float)(sum_ux[iy] / count[iy]) : 0.0f;
        float uy_m = (count[iy] > 0)
                     ? (float)(sum_uy[iy] / count[iy]) : 0.0f;
        float of   = (float)(obs_frac[iy] / p.Nx);

        f << iy   << ","
          << ux_m << ","
          << uy_m << ","
          << of   << "\n";
    }

    f.close();
}

// ── Checkpoint ───────────────────────────────────────────────────
void save_checkpoint(const std::string& fpath,
                      const float* d_f,
                      const SimParams& p,
                      int timestep)
{
    int N = p.Nx * p.Ny * Q;
    std::vector<float> h_f(N);
    cudaMemcpy(h_f.data(), d_f,
               N*sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream f(fpath, std::ios::binary);
    // Header: timestep + Nx + Ny
    f.write(reinterpret_cast<const char*>(&timestep), sizeof(int));
    f.write(reinterpret_cast<const char*>(&p.Nx),     sizeof(int));
    f.write(reinterpret_cast<const char*>(&p.Ny),     sizeof(int));
    // Datos
    f.write(reinterpret_cast<const char*>(h_f.data()),
            N * sizeof(float));
    f.close();
}

bool load_checkpoint(const std::string& fpath,
                      float* d_f,
                      const SimParams& p,
                      int& timestep)
{
    std::ifstream f(fpath, std::ios::binary);
    if (!f.is_open()) return false;

    int t_saved, Nx_saved, Ny_saved;
    f.read(reinterpret_cast<char*>(&t_saved),  sizeof(int));
    f.read(reinterpret_cast<char*>(&Nx_saved), sizeof(int));
    f.read(reinterpret_cast<char*>(&Ny_saved), sizeof(int));

    // Verificar compatibilidad
    if (Nx_saved != p.Nx || Ny_saved != p.Ny) {
        std::cerr << "❌ Checkpoint incompatible: "
                  << Nx_saved << "x" << Ny_saved
                  << " vs " << p.Nx << "x" << p.Ny << "\n";
        return false;
    }

    int N = p.Nx * p.Ny * Q;
    std::vector<float> h_f(N);
    f.read(reinterpret_cast<char*>(h_f.data()), N * sizeof(float));
    f.close();

    cudaMemcpy(d_f, h_f.data(),
               N*sizeof(float), cudaMemcpyHostToDevice);

    timestep = t_saved;
    std::cout << "✔ Checkpoint cargado desde t=" << timestep << "\n";
    return true;
}