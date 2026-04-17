# ================================================================
# Makefile
# Compila el simulador LBM D2Q9 en CUDA.
# Uso:
#   make        → compila todo
#   make clean  → borra compilados
#   make test   → compila y corre un caso de prueba
# ================================================================

# Compilador y flags
NVCC     = nvcc
CFLAGS   = -O3 -std=c++17 -arch=sm_75
# sm_75 = Tesla T4 (Colab gratuito)
# sm_80 = A100 (Colab Pro)
# Si no sabes tu GPU: nvidia-smi --query-gpu=compute_cap --format=csv

# Directorios
CUDA_DIR    = cuda
KERNEL_DIR  = $(CUDA_DIR)/kernel
IO_DIR      = $(CUDA_DIR)/io

# Archivos fuente
SOURCES = $(CUDA_DIR)/main.cu \
          $(KERNEL_DIR)/macro.cu \
          $(KERNEL_DIR)/collision.cu \
          $(KERNEL_DIR)/boundary.cu \
          $(KERNEL_DIR)/streaming.cu \
          $(IO_DIR)/write_vti.cu \
          $(IO_DIR)/write_csv.cu

# Ejecutable
TARGET = $(CUDA_DIR)/lbm_sim

# Regla principal
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) $(SOURCES) -o $(TARGET)
	@echo "✔ Compilación exitosa: $(TARGET)"

# Limpiar
clean:
	rm -f $(TARGET)
	@echo "✔ Limpieza completa"

# Compilar y correr caso de prueba
test: $(TARGET)
	python3 pipeline/scaling.py
	python3 pipeline/mesh_gen.py
	@echo "✔ Test de Python completado"
	@echo "Para correr simulación: ./$(TARGET) config.json mask.bin 0"

.PHONY: all clean test