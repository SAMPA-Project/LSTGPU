#pragma once

#include "cuda/horizon_buffer.cuh"

/* Direct and diffuse shortwave of hour h of day n for every voxel: sun visibility from the horizon buffers
   (canopies transmit exp(-k LAI)), direct irradiance on the inclined voxel, and the modified anisotropic Perez
   diffuse model with the visible parts of the sky (cosine-weighted), the horizon band (front half, cosine-weighted)
   and the circumsolar region. Also sets the sky view factor and resets the reflected and surrounding-longwave terms
   of the hour.
   sun_circumsolar_region: per hour, the solid angle of every direction bin inside the circumsolar region. */
__global__ void shortwave_kernel(const grid_s_type* grid_s, const grid_s_type* grid_s_veg, const float2* grid_s_cover,
    const float2* grid_s_cover_veg, const float* grid_s_front_hz, const float* sun_circumsolar_region, float* targetData,
    int n, int h);
