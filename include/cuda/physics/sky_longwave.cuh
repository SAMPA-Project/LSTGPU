#pragma once

/* Longwave irradiance from the sky: the Brutsaert (1975) clear-sky emissivity, its cloud correction (EnergyPlus /
   ASHRAE, Walton), the daytime cloud fraction from the measured irradiance, and the tilt correction of the sky view. */

#include "cuda/device_globals.cuh"

__device__ __host__ inline float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

// Cloud correction of the clear-sky emissivity (EnergyPlus / ASHRAE, Walton), cloud cover N in tenths
__device__ __host__ inline float eps_sky_walton_tenths(float eps_cs, float N_tenths) {
    eps_cs = clamp01(eps_cs);
    N_tenths = fminf(fmaxf(N_tenths, 0.0f), 10.0f);
    const float N = N_tenths;
    const float N2 = N * N;
    const float N3 = N2 * N;
    float factor = 1.0f + 0.0224f * N - 0.0035f * N2 + 0.00028f * N3;
    return clamp01(eps_cs * factor);
}

__device__ __host__ inline float eps_sky_walton_cf(float eps_cs, float cf) {
    cf = clamp01(cf);
    return eps_sky_walton_tenths(eps_cs, 10.0f * cf);
}

__device__ __forceinline__ float clipf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

// x^1.5 for 0 <= x <= 1
__device__ __forceinline__ float pow1p5(float x) {
    return (x <= 0.0f) ? 0.0f : x * sqrtf(x);
}

/* Daytime cloud fraction from the diffuse fraction of the measured irradiance, against an empirical clear-sky
   diffuse fraction; -1 at night or for low irradiance (the cloud cover series is used then). */
__device__ __forceinline__ float cloud_fraction_from_components(float Bh, float Dh, float mu, float Gh_min = 20.0f) {
    if (!(mu > 0.0f)) return -1.0f;

    const float Gh = Bh + Dh;
    if (!(Gh > Gh_min)) return -1.0f;

    const float kd = Dh / Gh;

    float kd_clear = 0.05f + 0.25f * pow1p5(1.0f - clipf(mu, 0.0f, 1.0f));
    kd_clear = clipf(kd_clear, 0.05f, 0.35f);

    const float kd_overcast = 0.95f;
    const float denom = kd_overcast - kd_clear;
    if (!(denom > 1e-6f)) return -1.0f;

    float cf = (kd - kd_clear) / denom;
    return clamp01(cf);
}

// Brutsaert (1975) clear-sky emissivity from the vapour pressure pw (Pa) and the air temperature Ta (K)
__device__ __forceinline__ float brutsaert_emissivity(float pw, float Ta) {
    float e_hPa = pw * 0.01f;
    e_hPa = fmaxf(e_hPa, 0.5f);
    return P_g[P_BRUTSAERT] * powf(e_hPa / Ta, 1.0f / 7.0f);
}

// Sky view factor with the tilt correction (at most +1.8 %)
__device__ __forceinline__ float tilted_sky_view(float F_svf, float slope) {
    float slope_deg = slope * 180.0f / (float)PI;
    float beta_scaled = 0.01f * fminf(fmaxf(slope_deg, 0.0f), 90.0f);

    float k3_beta = 0.7629f * powf(beta_scaled, 4)
        - 2.2215f * powf(beta_scaled, 3)
        + 1.7483f * beta_scaled * beta_scaled
        + 0.0540f * beta_scaled;

    k3_beta = fminf(fmaxf(k3_beta, 0.0f), 0.2f);

    return fminf(fmaxf(F_svf + 0.09f * k3_beta, 0.0f), 1.0f);
}
