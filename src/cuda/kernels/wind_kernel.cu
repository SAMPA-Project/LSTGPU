#include "cuda/kernels/wind_kernel.cuh"

__global__ void wind_shelter_kernel(const grid_s_type* grid_s, float* targetData, int n, int h) {
    const unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= num_data) return;

    int ind = n * num_hours + h;

    // wind direction: meteorological (from which), degrees; the horizon buffers are binned by the azimuth of the
    // direction seen from the voxel, clockwise from north
    float windDir_deg = wind_dir[ind];

    // shelter from the sector the wind comes from: the horizon sector of an azimuth looks along a ray from the voxel
    // towards it
    float dir_to_deg = windDir_deg;
    if (dir_to_deg >= 360.0f) dir_to_deg -= 360.0f;
    if (dir_to_deg < 0.0f)    dir_to_deg += 360.0f;

    int idir = (int)floorf(dir_to_deg / (float)hres);
    if (idir < 0) idir = 0;
    if (idir >= hsize) idir = hsize - 1;

    // vertical index of the horizon in that sector
    grid_s_type val = horizon_get(t_id, (grid_s_type)idir, grid_s);

    // discrete elevation band to angle
    float alpha_max = 0.0f;
    if (val > 0) {
        float alpha_deg = (float)val * (float)vres;
        alpha_max = alpha_deg * (PI / 180.0f);
    }

    // terrain/building shelter from alpha_max
    const float alpha0 = 0.0f * (PI / 180.0f); // threshold angle
    const float gamma = P_g[P_SHELTER_GAMMA];
    float terrain_shelter = expf(-gamma * fmaxf(0.0f, alpha_max - alpha0));
    terrain_shelter = fminf(fmaxf(terrain_shelter, 0.0f), 1.0f);

    // vegetation sheltering is applied in the SEB (canopy wind reduction)
    float veg_shelter = 1.0f;

    float shelter = terrain_shelter * veg_shelter;

    targetData[WIND_ZD * num_data + t_id] = shelter;
}
