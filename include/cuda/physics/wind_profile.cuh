#pragma once

/* Local wind speed of a voxel: the measured wind scaled by a power-law profile to 2 m above the voxel, and its
   reduction inside a tree canopy. The sheltering by terrain and buildings is from wind_shelter_kernel. */

#include "cuda/device_globals.cuh"

// Wind speed (m s-1) 2 m above a voxel at height z (m; above the local ground with wind_height_agl 1), from the wind ws
// (m s-1) measured at the reference height
__device__ __forceinline__ float profile_wind_speed(float ws, float z) {
    return ws * powf(fmaxf(z + 2.0f, 1.0f) / fmaxf(P_g[P_WIND_ZREF], 1.0f), P_g[P_WIND_EXP]);
}

// Beer-Lambert wind reduction factor inside a canopy with the LAI of day n (0.2 .. 1)
__device__ __forceinline__ float canopy_wind_factor(int n) {
    float f_leaf = 1.0f - expf(-P_g[P_WIND_CAN_K] * fmaxf(light_ext_coeffs[n], 0.0f));
    return fminf(fmaxf(1.0f - P_g[P_WIND_CAN_RED] * f_leaf, 0.2f), 1.0f);
}
