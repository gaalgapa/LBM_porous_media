// ================================================================
// d2q9.cuh
// Define las constantes del retículo D2Q9.
// Convención de índices:
//   0:(0,0)  1:(+x,0)  2:(0,+y)  3:(-x,0)  4:(0,-y)
//   5:(+x,+y) 6:(-x,+y) 7:(-x,-y) 8:(+x,-y)
// ================================================================

#pragma once

#define Q 9

// Vectores de velocidad
__constant__ int CX[Q] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
__constant__ int CY[Q] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

// Pesos
__constant__ float W[Q] = {
    4.0f/9.0f,
    1.0f/9.0f,  1.0f/9.0f,  1.0f/9.0f,  1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// Índices opuestos para bounce-back
// opp[k] = dirección opuesta a k
__constant__ int OPP[Q] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

// Velocidad del sonido al cuadrado
#define CS2 (1.0f / 3.0f)