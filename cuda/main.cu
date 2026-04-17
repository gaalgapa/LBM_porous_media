// ================================================================
// main.cu
// Programa principal del simulador LBM D2Q9.
//
// Uso:
//   ./lbm_sim config.json mask.bin seed
//
// Orden del bucle principal:
//   1. macro_kernel    — rho, ux, uy desde f actual
//   2. collision_kernel — colisión BGK → fnew
//   3. boundary_kernel  — Zou/He + extrapolación + BB paredes
//   4. streaming_kernel — fnew → f del siguiente paso
//   5. Convergencia con thrust::reduce cada 1000 pasos
//   6. Checkpoint cada ckpt_every pasos
//   7. VTI + CSV al converger o cada save_every pasos
// ================================================================

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "d2q9.cuh"
#include "headers.cuh"

// ── Macro de verificación CUDA ───────────────────────────────────
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            std::cerr << "CUDA error " << __FILE__                   \
                      << ":" << __LINE__ << "  "                     \
                      << cudaGetErrorString(_e) << "\n";             \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while(0)

// ── Punteros globales de device ──────────────────────────────────
float* d_f;
float* d_fnew;
float* d_rho;
float* d_ux;
float* d_uy;
float* d_ux_prev;
float* d_uy_prev;
float* d_num;
float* d_den;
bool*  d_obstacle;

// ================================================================
// Lectura de config.json
// Parseo manual simple — sin dependencias externas
// ================================================================
SimParams load_config(const std::string& config_path,
                       const std::string& run_dir,
                       int seed)
{
    SimParams p;

    std::ifstream f(config_path);
    if (!f.is_open()) {
        std::cerr << "❌ No se pudo abrir: " << config_path << "\n";
        exit(EXIT_FAILURE);
    }

    std::string line;
    auto extract = [](const std::string& line,
                      const std::string& key) -> std::string {
        size_t pos = line.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = line.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < line.size() &&
               (line[pos]==' ' || line[pos]=='"')) pos++;
        size_t end = pos;
        while (end < line.size() &&
               line[end]!=',' && line[end]!='"' &&
               line[end]!='}' && line[end]!='\n') end++;
        return line.substr(pos, end - pos);
    };

    // Leer todo el archivo en un string
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Extraer valores
    auto get = [&](const std::string& key) -> float {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = content.find(":", pos) + 1;
        while (content[pos] == ' ') pos++;
        return std::stof(content.substr(pos));
    };

    auto get_int = [&](const std::string& key) -> int {
        return (int)get(key);
    };

    p.Nx           = get_int("Nx");
    p.Ny           = get_int("Ny");
    p.tau          = get("tau");
    p.nu_lbm       = get("nu_lbm");
    p.u_inlet_lbm  = get("u_inlet_lbm");
    p.rho0         = 1.0f;
    p.t_max        = get_int("t_max");
    p.max_diff     = get("max_diff");
    p.save_every   = get_int("save_every");
    p.ckpt_every   = get_int("ckpt_every");
    p.c_l          = get("c_l");
    p.c_t          = get("c_t");
    p.rho_phy      = get("rho_phy");
    p.mu_phy       = get("mu_phy");
    p.seed         = seed;

    strncpy(p.run_dir, run_dir.c_str(), sizeof(p.run_dir)-1);

    std::cout << "✔ Config cargado: Nx=" << p.Nx
              << " Ny=" << p.Ny
              << " tau=" << p.tau
              << " u_inlet=" << p.u_inlet_lbm << "\n";
    return p;
}

// ================================================================
// Inicialización
// ================================================================
void initialize(const SimParams& p,
                const std::string& mask_path,
                const std::string& ckpt_path)
{
    size_t sz_f   = (size_t)p.Nx * p.Ny * Q * sizeof(float);
    size_t sz_mac = (size_t)p.Nx * p.Ny * sizeof(float);
    size_t sz_obs = (size_t)p.Nx * p.Ny * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&d_f,        sz_f));
    CUDA_CHECK(cudaMalloc(&d_fnew,     sz_f));
    CUDA_CHECK(cudaMalloc(&d_rho,      sz_mac));
    CUDA_CHECK(cudaMalloc(&d_ux,       sz_mac));
    CUDA_CHECK(cudaMalloc(&d_uy,       sz_mac));
    CUDA_CHECK(cudaMalloc(&d_ux_prev,  sz_mac));
    CUDA_CHECK(cudaMalloc(&d_uy_prev,  sz_mac));
    CUDA_CHECK(cudaMalloc(&d_num,      sz_mac));
    CUDA_CHECK(cudaMalloc(&d_den,      sz_mac));
    CUDA_CHECK(cudaMalloc(&d_obstacle, sz_obs));

    // Cargar máscara de obstáculos
    std::vector<uint8_t> h_mask_raw(p.Nx * p.Ny);
    std::vector<uint8_t> h_obs(p.Nx * p.Ny, 0);

    std::ifstream mfile(mask_path, std::ios::binary);
    if (mfile.is_open()) {
        mfile.read(reinterpret_cast<char*>(h_mask_raw.data()),
                   p.Nx * p.Ny);
        mfile.close();
        for (int i = 0; i < p.Nx * p.Ny; i++)
            h_obs[i] = (h_mask_raw[i] != 0) ? 1 : 0;
        std::cout << "✔ Máscara cargada: " << mask_path << "\n";
    } else {
        std::cerr << "⚠️  Máscara no encontrada. Sin obstáculos.\n";
    }

    CUDA_CHECK(cudaMemcpy(d_obstacle, h_obs.data(),
                          sz_obs, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((p.Nx + block.x-1) / block.x,
              (p.Ny + block.y-1) / block.y);

    // Intentar cargar checkpoint
    int t_start = 0;
    bool from_ckpt = load_checkpoint(ckpt_path, d_f, p, t_start);

    if (!from_ckpt) {
        // Inicializar desde cero
        initialize_kernel<<<grid, block>>>(d_f, d_rho,
                                            d_ux, d_uy, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "✔ Inicialización desde cero.\n";
    }

    CUDA_CHECK(cudaMemcpy(d_fnew, d_f, sz_f,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux_prev, d_ux, sz_mac,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy_prev, d_uy, sz_mac,
                          cudaMemcpyDeviceToDevice));
}

// ================================================================
// Bucle principal
// ================================================================
void run(const SimParams& p,
         const std::string& results_dir,
         const std::string& ckpt_path,
         int seed)
{
    dim3 block(32, 8);
    dim3 grid((p.Nx + block.x-1) / block.x,
              (p.Ny + block.y-1) / block.y);

    size_t sz_mac = (size_t)p.Nx * p.Ny * sizeof(float);

    float diff      = 1.0e10f;
    int   t         = 0;
    bool  converged = false;

    std::cout << "\nIniciando simulación"
              << " (t_max=" << p.t_max
              << ", max_diff=" << p.max_diff << ")...\n";

    while (t < p.t_max && !converged) {

        // ── (1) Macroscópicos ────────────────────────────────────
        macro_kernel<<<grid, block>>>(d_f, d_rho, d_ux, d_uy,
                                       d_obstacle, p);

        // ── (2) Colisión ─────────────────────────────────────────
        collision_kernel<<<grid, block>>>(d_f, d_fnew, d_rho,
                                           d_ux, d_uy,
                                           d_obstacle, p);

        // ── (3) Frontera ─────────────────────────────────────────
        boundary_kernel<<<grid, block>>>(d_fnew, d_obstacle, p);

        // ── (4) Streaming ────────────────────────────────────────
        streaming_kernel<<<grid, block>>>(d_fnew, d_f,
                                           d_obstacle, p);

        CUDA_CHECK(cudaDeviceSynchronize());

        // ── (5) Convergencia cada 1000 pasos ────────────────────
        if (t % 1000 == 0) {

            // Guardar velocidad anterior
            CUDA_CHECK(cudaMemcpy(d_ux_prev, d_ux, sz_mac,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_uy_prev, d_uy, sz_mac,
                                  cudaMemcpyDeviceToDevice));

            // Macroscópicos actualizados para el error
            macro_kernel<<<grid, block>>>(d_f, d_rho, d_ux, d_uy,
                                           d_obstacle, p);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Error con thrust::reduce directamente en GPU
            error_kernel<<<grid, block>>>(d_ux, d_uy,
                                           d_ux_prev, d_uy_prev,
                                           d_num, d_den,
                                           d_obstacle, p);
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::device_ptr<float> p_num(d_num);
            thrust::device_ptr<float> p_den(d_den);

            float sum_num = thrust::reduce(thrust::device,
                                           p_num,
                                           p_num + p.Nx*p.Ny,
                                           0.0f);
            float sum_den = thrust::reduce(thrust::device,
                                           p_den,
                                           p_den + p.Nx*p.Ny,
                                           0.0f);

            diff = (sum_den > 1.0e-12f)
                   ? sqrtf(sum_num / sum_den)
                   : 1.0e10f;

            std::cout << "t=" << t
                      << "  diff=" << diff << "\n";

            if (diff <= p.max_diff && t > 0)
                converged = true;
        }

        // ── (6) Checkpoint ───────────────────────────────────────
        if (t % p.ckpt_every == 0 && t > 0)
            save_checkpoint(ckpt_path, d_f, p, t);

        // ── (7) Guardar VTI ──────────────────────────────────────
        if (t % p.save_every == 0 || converged) {
            std::string vti_path = results_dir + "/campos_t"
                                   + std::to_string(t) + ".vti";
            write_vti(vti_path, d_ux, d_uy, d_rho,
                      d_obstacle, p, t);
        }
    }

    // ── Guardar resultados finales ───────────────────────────────
    std::string metrics_path = results_dir + "/metrics.csv";
    std::string profile_path = results_dir + "/profile_seed"
                               + std::to_string(seed) + ".csv";

    write_metrics_csv(metrics_path, d_rho, d_ux,
                      p, seed, t, converged);
    write_profile_csv(profile_path, d_ux, d_uy,
                      d_obstacle, p, seed);

    // VTI final
    std::string vti_final = results_dir + "/campos_final_seed"
                            + std::to_string(seed) + ".vti";
    write_vti(vti_final, d_ux, d_uy, d_rho, d_obstacle, p, t);

    std::cout << "\n✔ Simulación finalizada. t=" << t << "\n";
    if (converged)
        std::cout << "🎉 Convergida. diff=" << diff << "\n";
    else
        std::cout << "⚠️  t_max alcanzado sin converger.\n";
}

// ================================================================
// main
// ================================================================
int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Uso: ./lbm_sim config.json mask.bin seed\n";
        return EXIT_FAILURE;
    }

    std::string config_path = argv[1];
    std::string mask_path   = argv[2];
    int         seed        = std::stoi(argv[3]);

    // Directorio de resultados = directorio del config
    std::string run_dir = config_path.substr(
        0, config_path.find_last_of("/\\"));
    std::string results_dir = run_dir + "/results";
    std::string ckpt_path   = run_dir + "/checkpoint_seed"
                              + std::to_string(seed) + ".bin";

    // Verificar GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "❌ No se encontró GPU CUDA.\n";
        return EXIT_FAILURE;
    }

    // Cargar parámetros
    SimParams p = load_config(config_path, run_dir, seed);

    // Inicializar
    initialize(p, mask_path, ckpt_path);

    // Correr
    run(p, results_dir, ckpt_path, seed);

    // Liberar memoria
    cudaFree(d_f);       cudaFree(d_fnew);
    cudaFree(d_rho);     cudaFree(d_ux);
    cudaFree(d_uy);      cudaFree(d_ux_prev);
    cudaFree(d_uy_prev); cudaFree(d_num);
    cudaFree(d_den);     cudaFree(d_obstacle);

    return EXIT_SUCCESS;
}