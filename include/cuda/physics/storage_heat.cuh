#pragma once

/* Storage heat flux: the objective hysteresis model (OHM) of the material and the calibrated seasonal storage term. */

#include "cuda/device_globals.cuh"

// Hysteresis part a1 Q* + a2 dQ* of material cc; dQ_net: change of the net radiation over the last hour (W m-2)
__device__ __forceinline__ float ohm_storage(int cc, float Q_net, float dQ_net) {
    return class_matrix[cc][M_OHM_A1] * Q_net + class_matrix[cc][M_OHM_A2] * dQ_net;
}

// Seasonal storage term (W m-2) of day n: extra storage (cooling) in winter, less in summer
__device__ __forceinline__ float seasonal_storage(int n) {
    return P_g[P_SEAS_AMP] * cosf(2.0f * PI * ((float)n - P_g[P_SEAS_PHASE]) / 365.0f);
}
