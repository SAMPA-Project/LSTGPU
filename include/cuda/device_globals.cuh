#pragma once

/* __constant__ / __device__ symbols shared by every kernel translation unit. They are defined exactly once, in
   src/cuda/device_globals.cu; every .cu file that uses them includes this header and relies on CUDA
   relocatable device code to resolve them (CMakeLists.txt: CUDA_SEPARABLE_COMPILATION ON). The host uploads
   them in src/cuda/device_upload.cu (model, forcing) and src/cuda/lst_pipeline.cu (raster and bins). */

#include "lst_constants.h"
#include "model_params.h"

// ---- raster and voxels ----
extern __constant__ int width, height;     // raster size (cells)
extern __constant__ float gridRes;         // cell size (m)
extern __constant__ float maxz;            // highest top surface + 2 cells: rays above it can't hit anything
extern __constant__ unsigned int num_data; // number of voxels (top + under-canopy + wall)

// ---- time ----
extern __constant__ int num_hours;     // hours per day in the forcing series
extern __constant__ float time_inter;  // time step (h)

// ---- hemisphere direction bins ----
extern __constant__ float hres, vres, vres_f;        // bin size (degrees): azimuth, elevation (upper hemisphere), elevation (full sphere)
extern __constant__ unsigned int hsize, vsize, vsize_f; // bin counts: 360 / hres, 90 / vres, 180 / vres_f
extern __constant__ unsigned int bits_v;             // bits per horizon-buffer entry: ceil(log2(vsize))
extern __constant__ float total_view_cover;          // cosine-weighted solid angle of the upper hemisphere of a horizontal plane (~pi)
extern __constant__ float total_horizont_cover;      // solid angle of the lowest elevation band (not used by the kernels)

extern __device__ unsigned int max_indices_g;   // slots per voxel of the viewshed cache
extern __device__ unsigned int g_max_unique_in; // highest slot in use, found while building the caches

// ---- hourly forcing, indexed day * num_hours + hour ----
extern __device__ float2 sun_azm_alt[FORCING_ARRAY_SIZE]; // sun azimuth and altitude (degrees)
extern __device__ float direct_irad_meas[FORCING_ARRAY_SIZE], diffuse_irad_meas[FORCING_ARRAY_SIZE]; // horizontal (W m-2)
extern __device__ float ambient_temperature[FORCING_ARRAY_SIZE]; // degC
extern __device__ float p_w[FORCING_ARRAY_SIZE];                 // vapour pressure (Pa)
extern __device__ float wind_speed[FORCING_ARRAY_SIZE], wind_dir[FORCING_ARRAY_SIZE]; // m s-1, degrees (from)
extern __device__ float c_f[FORCING_ARRAY_SIZE];                 // cloud cover fraction
extern __device__ float precip_meas[FORCING_ARRAY_SIZE];         // mm per time step
extern __device__ float light_ext_coeffs[365];                   // LAI per day
extern __device__ float E0[365];                                 // extraterrestrial irradiance (W m-2)

// ---- model ----
extern __device__ float F_matrix[6][8];                       // Perez coefficients
extern __device__ float class_matrix[NUM_MATERIALS][M_NUM];   // material table
extern __device__ float P_g[P_NUM];                           // model constants
