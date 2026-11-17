#pragma once

#include "cuda/device_globals.cuh"

/* Exchange with the surrounding surfaces of hour h of day n, over every voxel's viewshed cache:
     reflect_order 0: reflected shortwave of first order (from the sources' direct + diffuse) and longwave
                      emitted by the sources at their LST of the previous hour (Lsurr),
     reflect_order 1: reflected shortwave of second order (from the sources' first-order reflection).
   The view factors of the cached sources are rescaled to the obstructed view, 1 - SVF. */
__global__ void surroundings_kernel(float* targetData, int n, int h, const int* svf_ind, bool reflect_order);
