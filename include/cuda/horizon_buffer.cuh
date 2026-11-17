#pragma once

/* Quantised horizon buffers: for every voxel and azimuth sector, the highest obstructed elevation band
   (0 .. vsize - 1) in bits_v = ceil(log2(vsize)) bits. One buffer for terrain and buildings (grid_s), one for
   vegetation (grid_s_veg). Words are stored word-major, word w of voxel t at grid_s[t + w * num_data], so
   neighbouring threads read neighbouring words. */

#include "cuda/device_globals.cuh"

typedef unsigned int grid_s_type;
#define define_grid_s_bits 32

#define get_bit_grid_s(t_id, loc, grid_ss) ( \
  (((grid_ss)[(t_id) + (grid_s_type)((loc) / define_grid_s_bits) * (grid_s_type)num_data] >> ((loc) % define_grid_s_bits)) & 1ULL) \
)

#define set_bit_grid_s(t_id, loc, grid_ss) ( \
  ( (grid_ss)[(t_id) + (grid_s_type)((loc) / define_grid_s_bits) * (grid_s_type)num_data] |= (1ULL << ((loc) % define_grid_s_bits)) ) \
)

#define clear_bit_grid_s(t_id, loc, grid_ss) ( \
  ( (grid_ss)[(t_id) + (grid_s_type)((loc) / define_grid_s_bits) * (grid_s_type)num_data] &= ~(1ULL << ((loc) % define_grid_s_bits)) ) \
)

// Highest obstructed elevation band of voxel t_id in azimuth sector i.
__device__ inline grid_s_type horizon_get(unsigned int t_id, grid_s_type i, const grid_s_type* grid_s) {
    grid_s_type base = i * (grid_s_type)bits_v;
    grid_s_type val = 0;
    for (grid_s_type b = 0; b < (grid_s_type)bits_v; ++b)
        if (get_bit_grid_s(t_id, base + b, grid_s)) val |= ((grid_s_type)1) << b;
    return val;
}

// Raises the stored band of voxel t_id in sector i to `val` if it is higher.
__device__ inline void horizon_raise(unsigned int t_id, grid_s_type i, grid_s_type val, grid_s_type* grid_s) {
    grid_s_type base = i * (grid_s_type)bits_v;
    if (horizon_get(t_id, i, grid_s) < val) {
        for (grid_s_type b = 0; b < (grid_s_type)bits_v; ++b) clear_bit_grid_s(t_id, base + b, grid_s);
        for (grid_s_type b = 0; b < (grid_s_type)bits_v; ++b)
            if (val & (((grid_s_type)1) << b)) set_bit_grid_s(t_id, base + b, grid_s);
    }
}
