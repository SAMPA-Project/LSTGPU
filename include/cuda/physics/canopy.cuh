#pragma once

/* Beer-Lambert attenuation of radiation by a tree canopy with the LAI of day n. */

#include "cuda/device_globals.cuh"

// Canopy fraction 1 - exp(-k LAI): the share of the radiation the canopy intercepts.
__device__ __forceinline__ float canopy_fraction(int n) {
    return 1.0f - expf(-P_g[P_CANOPY_K] * fmaxf(light_ext_coeffs[n], 0.0f));
}

// Canopy transmittance exp(-k LAI).
__device__ __forceinline__ float canopy_transmittance(int n) {
    return expf(-P_g[P_CANOPY_K] * fmaxf(light_ext_coeffs[n], 0.0f));
}
