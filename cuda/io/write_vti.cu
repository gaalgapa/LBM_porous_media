// ================================================================
// io/write_vti.cu
// Escribe campos de velocidad, densidad y obstáculos en formato
// VTK ImageData (.vti) para visualización en ParaView.
// Guarda en unidades físicas (m/s, Pa) no en lattice units.
// ================================================================

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "../headers.cuh"

// ── Utilidad: convertir array device a host ──────────────────────
static void d2h_float(const float* d_ptr, std::vector<float>& h_vec,
                       int n)
{
    h_vec.resize(n);
    cudaMemcpy(h_vec.data(), d_ptr,
               n * sizeof(float), cudaMemcpyDeviceToHost);
}

static void d2h_uint8(const uint8_t* d_ptr, std::vector<uint8_t>& h_vec, int n) // <-- Cambio a uint8_t*
{
    h_vec.resize(n);
    cudaMemcpy(h_vec.data(), d_ptr, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}

// ── Función principal ────────────────────────────────────────────
void write_vti(const std::string& fpath,
               const float* d_ux,
               const float* d_uy,
               const float* d_rho,
               const uint8_t*  d_obstacle,
               const SimParams& p,
               int timestep)
{
    int N = p.Nx * p.Ny;

    // Copiar campos de GPU a CPU
    std::vector<float>   h_ux(N), h_uy(N), h_rho(N);
    std::vector<uint8_t> h_obs(N);

    d2h_float(d_ux,       h_ux,  N);
    d2h_float(d_uy,       h_uy,  N);
    d2h_float(d_rho,      h_rho, N);
    d2h_uint8(d_obstacle, h_obs, N);

    // Convertir a unidades físicas
    // ux, uy: [lu/ts] → [m/s]
    // rho: [lu] → presión [Pa] via p = rho * cs² * (rho_phy * c_l² / c_t²)
    float vel_scale = p.c_l / p.c_t;
    float pre_scale = p.rho_phy * (p.c_l * p.c_l) /
                      (p.c_t  * p.c_t) / 3.0f;

    std::vector<float> ux_phy(N), uy_phy(N), pre_phy(N);
    for (int i = 0; i < N; i++) {
        ux_phy[i]  = h_ux[i]  * vel_scale;
        uy_phy[i]  = h_uy[i]  * vel_scale;
        pre_phy[i] = h_rho[i] * pre_scale;
    }

    // Escribir VTI
    // Spacing usa c_l para que ParaView trabaje en metros
    std::ofstream f(fpath);

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"ImageData\" version=\"0.1\" "
      << "byte_order=\"LittleEndian\">\n";
    f << "  <ImageData WholeExtent=\""
      << "0 " << p.Nx-1 << " "
      << "0 " << p.Ny-1 << " "
      << "0 0\" "
      << "Origin=\"0 0 0\" "
      << "Spacing=\"" << p.c_l << " " << p.c_l << " 1\">\n";
    f << "    <Piece Extent=\""
      << "0 " << p.Nx-1 << " "
      << "0 " << p.Ny-1 << " "
      << "0 0\">\n";
    f << "      <PointData>\n";

    // ── ux ───────────────────────────────────────────────────────
    f << "        <DataArray type=\"Float32\" Name=\"ux_ms\" "
      << "format=\"ascii\">\n         ";
    for (int ix = 0; ix < p.Nx; ix++)
        for (int iy = 0; iy < p.Ny; iy++)
            f << " " << ux_phy[ix * p.Ny + iy];
    f << "\n        </DataArray>\n";

    // ── uy ───────────────────────────────────────────────────────
    f << "        <DataArray type=\"Float32\" Name=\"uy_ms\" "
      << "format=\"ascii\">\n         ";
    for (int ix = 0; ix < p.Nx; ix++)
        for (int iy = 0; iy < p.Ny; iy++)
            f << " " << uy_phy[ix * p.Ny + iy];
    f << "\n        </DataArray>\n";

    // ── presión ──────────────────────────────────────────────────
    f << "        <DataArray type=\"Float32\" Name=\"pressure_Pa\" "
      << "format=\"ascii\">\n         ";
    for (int ix = 0; ix < p.Nx; ix++)
        for (int iy = 0; iy < p.Ny; iy++)
            f << " " << pre_phy[ix * p.Ny + iy];
    f << "\n        </DataArray>\n";

    // ── magnitud de velocidad ────────────────────────────────────
    f << "        <DataArray type=\"Float32\" Name=\"u_mag_ms\" "
      << "format=\"ascii\">\n         ";
    for (int ix = 0; ix < p.Nx; ix++)
        for (int iy = 0; iy < p.Ny; iy++) {
            int i = ix * p.Ny + iy;
            f << " " << sqrtf(ux_phy[i]*ux_phy[i] +
                               uy_phy[i]*uy_phy[i]);
        }
    f << "\n        </DataArray>\n";

    // ── obstáculo ────────────────────────────────────────────────
    f << "        <DataArray type=\"UInt8\" Name=\"obstacle\" "
      << "format=\"ascii\">\n         ";
    for (int ix = 0; ix < p.Nx; ix++)
        for (int iy = 0; iy < p.Ny; iy++)
            f << " " << (int)h_obs[ix * p.Ny + iy];
    f << "\n        </DataArray>\n";

    f << "      </PointData>\n";
    f << "    </Piece>\n";
    f << "  </ImageData>\n";
    f << "</VTKFile>\n";

    f.close();
}