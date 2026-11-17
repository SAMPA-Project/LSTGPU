#pragma once

/* Convective heat transfer coefficient h_c (W m-2 K-1) of the sensible heat flux QH = h_c (Ts - Ta): forced
   convection (McAdams/Juerges) blended by the surface tilt, natural convection over a warmer (unstable) or cooler
   (stable) surface, and their combination. cos_phi = |cos(slope)|: 1 = horizontal, 0 = vertical. */

#include "cuda/device_globals.cuh"

// Forced convection at the local wind speed Vf (m s-1)
__device__ __forceinline__ float forced_convection(float Vf, float cos_phi) {
    float h_c_for_horizontal;
    if (Vf < 5.0f) {
        h_c_for_horizontal = 5.7f + 3.8f * Vf;
    }
    else {
        h_c_for_horizontal = 7.14f * powf(Vf, 0.78f);
    }
    float h_c_for_vertical = 5.7f + 3.8f * Vf;
    return cos_phi * h_c_for_horizontal + (1.0f - cos_phi) * h_c_for_vertical;
}

// Natural convection over a surface warmer than the air by delta_T (K)
__device__ __forceinline__ float natural_convection_unstable(float delta_T, float cos_phi) {
    return 9.482f * powf(delta_T, 1.0f / 3.0f) / (7.283f - cos_phi);
}

// Natural convection over a surface cooler than the air by delta_T (K)
__device__ __forceinline__ float natural_convection_stable(float delta_T, float cos_phi) {
    return 1.810f * powf(delta_T, 1.0f / 3.0f) / (1.382f + cos_phi);
}

// Natural convection at the surface temperature Ts and the air temperature Ta (K); 0 at Ts = Ta
__device__ __forceinline__ float natural_convection(float Ts, float Ta, float cos_phi) {
    const float delta_T = fabsf(Ts - Ta);
    if (Ts > Ta) return natural_convection_unstable(delta_T, cos_phi);
    if (Ts < Ta) return natural_convection_stable(delta_T, cos_phi);
    return 0.0f;
}

// h_c from the forced and natural parts, floored at hc_min and scaled by the calibrated hc_scale
__device__ __forceinline__ float heat_transfer_coefficient(float h_c_for, float h_c_nat) {
    float h_c = sqrtf(h_c_for * h_c_for + h_c_nat * h_c_nat);
    return P_g[P_HC_SCALE] * fmaxf(h_c, P_g[P_HC_MIN]);
}
