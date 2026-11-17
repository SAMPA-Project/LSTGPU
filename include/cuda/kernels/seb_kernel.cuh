#pragma once

#include "cuda/device_globals.cuh"

/* Surface energy balance of hour h of day n for every voxel, solved for the LST by a damped Newton iteration:
     (1 - albedo) I + emissivity (Lsky + Lsurr) = emissivity sigma Ts^4 + QH + QE + QS
   with the sky longwave (Brutsaert clear-sky emissivity, cloud correction; the canopy emits at air temperature in the
   part of the sky seen through it, grid_s_cover_veg), forced and natural convection (h_c from the sheltered local
   wind), OHM storage heat flux with an empirical seasonal term, and Priestley-Taylor latent heat limited by a soil
   water bucket. start_h is the first simulated hour of a day. */
__global__ void seb_kernel(const unsigned int* grid, const float2* grid_s_cover_veg, float* targetData, int n, int h, int start_h);
