#include "cuda/device_globals.cuh"

// ---------------- shared device globals (declared extern in device_globals.cuh) ----------------

__constant__ int width, height;
__constant__ float gridRes;
__constant__ float maxz;
__constant__ unsigned int num_data;

__constant__ int num_hours;
__constant__ float time_inter;

__constant__ float hres, vres, vres_f;
__constant__ unsigned int hsize, vsize, vsize_f;
__constant__ unsigned int bits_v;
__constant__ float total_view_cover;
__constant__ float total_horizont_cover;

__device__ unsigned int max_indices_g;
__device__ unsigned int g_max_unique_in;

__device__ float2 sun_azm_alt[FORCING_ARRAY_SIZE];
__device__ float direct_irad_meas[FORCING_ARRAY_SIZE], diffuse_irad_meas[FORCING_ARRAY_SIZE];
__device__ float ambient_temperature[FORCING_ARRAY_SIZE];
__device__ float p_w[FORCING_ARRAY_SIZE];
__device__ float wind_speed[FORCING_ARRAY_SIZE], wind_dir[FORCING_ARRAY_SIZE];
__device__ float c_f[FORCING_ARRAY_SIZE];
__device__ float precip_meas[FORCING_ARRAY_SIZE];
__device__ float light_ext_coeffs[365];
__device__ float E0[365];

__device__ float F_matrix[6][8];
__device__ float class_matrix[NUM_MATERIALS][M_NUM];
__device__ float P_g[P_NUM];
