#pragma once

#include "cuda/horizon_buffer.cuh"

/* Wind shelter factor of hour h of day n for every voxel, exp(-gamma alpha_max), where alpha_max is the
   upwind horizon elevation read from the terrain/building horizon buffer (Winstral & Marks style metric). */
__global__ void wind_shelter_kernel(const grid_s_type* grid_s, float* targetData, int n, int h);
