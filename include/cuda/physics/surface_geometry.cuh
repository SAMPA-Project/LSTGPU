#pragma once

/* Orientation of a voxel's surface from its slope and aspect (radians, aspect clockwise from north), in the frame of
   the rays: x east, y south, z up. */

#include "cuda/device_globals.cuh"

// Unit normal of the surface.
__device__ __forceinline__ float3 surface_normal(float slope, float aspect) {
    float3 nrm;
    nrm.x = sinf(slope) * sinf(aspect);
    nrm.y = -sinf(slope) * cosf(aspect);
    nrm.z = cosf(slope);

    const float len = sqrtf(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
    nrm.x /= len;
    nrm.y /= len;
    nrm.z /= len;
    return nrm;
}

// Cosine of the angle of incidence of the sun on the surface; cos_z: cosine of the solar zenith angle.
__device__ __forceinline__ float cos_incidence(float cos_z, float slope, float aspect, float sun_azimuth_deg) {
    return cos_z * cosf(slope) + sinf(acosf(cos_z)) * sinf(slope) * cosf(sun_azimuth_deg * DTR - (aspect));
}
