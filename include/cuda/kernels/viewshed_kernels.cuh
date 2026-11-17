#pragma once

#include "cuda/horizon_buffer.cuh"

/* One hemisphere direction (azimuth bin i, elevation bin j of the full sphere) for every voxel: marches a ray
   from the voxel over the height raster (2D DDA) and records
     - the quantised horizon in grid_s (terrain/buildings) and grid_s_veg (vegetation),
     - the visible sky as cosine-weighted solid angle (bin solid angle x cos to the voxel's normal) of the directions
       whose ray escapes, in grid_s_cover (.x: whole hemisphere, .y: lowest band), the part of it seen through the
       canopy in grid_s_cover_veg, and the cosine-weighted lowest band in front of the surface in grid_s_front_hz,
     - the index of the first voxel hit (a top or wall voxel; grid_t maps the raster cell the ray lands in to its
       top voxel) in the voxel's viewshed cache svf_ind (slot-major, unique
       entries; negative if the ray passed through vegetation first; with lsurr_ray_weights the hit count).
   Called once per direction, before the time loop. */
__global__ void viewshed_kernel(float patch_area, grid_s_type* grid_s, grid_s_type* grid_s_veg, float2* grid_s_cover,
    float2* grid_s_cover_veg, float* grid_s_front_hz, const unsigned int* grid_w, float3 Cdir, const unsigned int* grid,
    const float* targetData, int i, int j, int* svf_ind, const unsigned int* grid_t);
