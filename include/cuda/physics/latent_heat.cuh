#pragma once

/* Latent heat flux of pervious surfaces: Priestley & Taylor (1972) with the FAO-56 slope of the saturation vapour
   pressure curve. The limits by the available water (soil water bucket, water stress) are applied in seb_kernel. */

#include "cuda/device_globals.cuh"

// Slope of the saturation vapour pressure curve (kPa K-1) at the air temperature Ta_c (degC), FAO-56
__device__ __forceinline__ float saturation_slope(float Ta_c) {
    float es_kPa = 0.6108f * expf(17.27f * Ta_c / (Ta_c + 237.3f));
    return 4098.0f * es_kPa / powf(Ta_c + 237.3f, 2.0f);
}

// Latent heat of vaporization (J kg-1) at the air temperature Ta (K)
__device__ __forceinline__ float latent_heat_vaporization(float Ta) {
    return 2.501e6f - 2370.0f * (Ta - 273.15f);
}

// Priestley-Taylor latent heat flux (W m-2) from the available energy (W m-2); gamma: psychrometric constant (kPa K-1)
__device__ __forceinline__ float priestley_taylor(float alpha, float Delta, float gamma, float avail_energy) {
    return alpha * (Delta / (Delta + gamma)) * avail_energy;
}
