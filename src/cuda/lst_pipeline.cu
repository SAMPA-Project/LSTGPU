/* The GPU pipeline: uploads the voxels, the model and the forcing (device_upload.cu), builds the horizon buffers
   and viewshed caches once (one ray per hemisphere direction and voxel, viewshed_kernels.cu), then steps through
   the hours: shortwave (irradiance_kernel.cu), two orders of reflection with the surrounding longwave
   (reflection_kernels.cu), wind sheltering (wind_kernel.cu), and the surface energy balance (seb_kernel.cu). */

#include "lst_gpu_api.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "sky_bins.h"
#include "cuda/device_globals.cuh"
#include "cuda/device_upload.cuh"
#include "cuda/horizon_buffer.cuh"
#include "cuda/kernels/irradiance_kernel.cuh"
#include "cuda/kernels/reflection_kernels.cuh"
#include "cuda/kernels/seb_kernel.cuh"
#include "cuda/kernels/viewshed_kernels.cuh"
#include "cuda/kernels/wind_kernel.cuh"

#define CUDA_BLOCK_SIZE 128

namespace lstgpu {

namespace {

void print_gpu_memory(const char* label) {
    size_t free_m, total_m;
    cudaMemGetInfo(&free_m, &total_m);
    std::cout << label << ": " << free_m / (1024 * 1024) << " MiB free of " << total_m / (1024 * 1024) << " MiB\n";
}

} // namespace

int run_lst_gpu(const SurfaceModel& sm, const Forcing& forcing, const std::vector<SunSample>& sun,
                const ModelParams& mp, const RunSettings& rs, const HourCallback& on_hour) {
    cudaDeviceProp prop;
    cudaSafeCall(cudaSetDevice(0));
    cudaSafeCall(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    const unsigned int num_b = (unsigned int)sm.voxels.size();
    if (num_b > SVF_ID_MASK) { // the viewshed cache packs the voxel index into 23 bits
        std::cerr << "Error: " << num_b << " voxels, more than the 2^23 the viewshed caches can index; use a smaller tile\n";
        return 1;
    }
    const int W = sm.width, H = sm.height;
    const float h_hres = rs.hres, h_vres = rs.vres, h_vres_f = rs.vres;
    const float h_gridRes = sm.res;

    unsigned int h_hsize = (int)ceil(360.0f / (float)h_hres);
    unsigned int h_vsize = (int)ceil(90.0f / (float)h_vres);
    unsigned int h_vsize_f = (int)ceil(180.0f / (float)h_vres_f);

    // horizon buffer words per voxel, and bits per entry: ceil(log2(h_vsize))
    unsigned int hgrid_size_s = ceilf(((float)h_hsize * ceilf(log2(h_vsize))) / define_grid_s_bits);
    unsigned int hbits_v = (unsigned int)ceilf(log2f((float)h_vsize));

    // ---------------- raster: packed heights and the wall / under-canopy voxels per cell ----------------
    unsigned int *grid, *grid_w;
    unsigned int* grid_t; // raster cell -> its top voxel, GRID_NO_TARGET where the cell has none
    grid_s_type *grid_s, *grid_s_veg;
    float2 *grid_s_cover, *grid_s_cover_veg; // cosine-weighted visible sky (.x hemisphere, .y lowest band), through the canopy
    float* grid_s_front_hz;                  // cosine-weighted lowest band in front of the surface
    float* gridcells_data_d;
    int* svf_ind = 0;

    cudaSafeCall(cudaMalloc((void**)&grid_w, W * H * sizeof(unsigned int) * 2));
    cudaSafeCall(cudaMalloc((void**)&grid_t, W * H * sizeof(unsigned int)));
    cudaSafeCall(cudaMalloc((void**)&grid, W * H * sizeof(unsigned int)));

    const size_t grid_s_bytes = (size_t)hgrid_size_s * num_b * sizeof(grid_s_type);
    cudaSafeCall(cudaMalloc((void**)&grid_s, grid_s_bytes));
    cudaSafeCall(cudaMemset(grid_s, 0, grid_s_bytes));
    cudaSafeCall(cudaMalloc((void**)&grid_s_veg, grid_s_bytes));
    cudaSafeCall(cudaMemset(grid_s_veg, 0, grid_s_bytes));

    cudaSafeCall(cudaMalloc((void**)&grid_s_cover, num_b * sizeof(float2)));
    cudaSafeCall(cudaMemset(grid_s_cover, 0, num_b * sizeof(float2)));
    cudaSafeCall(cudaMalloc((void**)&grid_s_cover_veg, num_b * sizeof(float2)));
    cudaSafeCall(cudaMemset(grid_s_cover_veg, 0, num_b * sizeof(float2)));
    cudaSafeCall(cudaMalloc((void**)&grid_s_front_hz, num_b * sizeof(float)));
    cudaSafeCall(cudaMemset(grid_s_front_hz, 0, num_b * sizeof(float)));

    cudaSafeCall(cudaMalloc((void**)&gridcells_data_d, (size_t)NUM_ATTRIBUTES * num_b * sizeof(float)));

    // cosine-weighted solid angle of the hemisphere, which normalises the sky view factor, and of its lowest band
    float total_area, total_h_area;
    hemisphere_cover(h_hsize, h_vsize, h_hres, h_vres, total_area, total_h_area);
    cudaSafeCall(cudaMemcpyToSymbol(total_view_cover, (void*)&total_area, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(total_horizont_cover, (void*)&total_h_area, sizeof(float), 0, cudaMemcpyHostToDevice));

    /* Per raster cell: number of wall and under-canopy voxels and the index of the first, and the raster position
       (GRID_POS) of every voxel, by the voxels' own indices (stored under-canopy first, then walls). */
    std::vector<float> voxel_pos(num_b);
    float maxz_h = -9999;
    {
        std::vector<unsigned int> h_grid_w((size_t)W * H * 2, 0);
        std::vector<unsigned int> h_grid_t((size_t)W * H, GRID_NO_TARGET);
        for (int v = 0; v < sm.num_top; ++v) {
            voxel_pos[v] = (float)(sm.voxels[v].x + sm.voxels[v].y * W);
            if (sm.voxels[v].z > maxz_h) maxz_h = sm.voxels[v].z;
        }
        for (int x = 0; x < W; ++x)
            for (int y = 0; y < H; ++y) {
                const int c = sm.cell(x, y);
                if (sm.top_voxel[c] == -1) continue;
                h_grid_t[(size_t)x + (size_t)y * W] = (unsigned int)sm.top_voxel[c];
                const size_t index = 2 * ((size_t)x + (size_t)y * W);
                h_grid_w[index + 0] = (unsigned int)sm.extra_voxels[c].size();
                h_grid_w[index + 1] = sm.extra_voxels[c].empty() ? 0u : (unsigned int)sm.extra_voxels[c][0];
                for (size_t w = 0; w < sm.extra_voxels[c].size(); ++w)
                    voxel_pos[sm.extra_voxels[c][w]] = (float)(x + y * W);
            }
        cudaSafeCall(cudaMemcpy(grid_w, h_grid_w.data(), h_grid_w.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(grid_t, h_grid_t.data(), h_grid_t.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    maxz_h += h_gridRes * 2.0f;

    // initial voxel state
    {
        std::vector<float> state((size_t)NUM_ATTRIBUTES * num_b, 0.0f);
        for (unsigned int i = 0; i < num_b; ++i) {
            const Voxel& v = sm.voxels[i];
            state[SURFACE_TEMPERATURE_H * (size_t)num_b + i] = 9999.0f;
            state[SLOPE * (size_t)num_b + i] = v.slope;
            state[ASPECT * (size_t)num_b + i] = v.aspect;
            state[CLASSTYPE * (size_t)num_b + i] = v.material;
            state[CELL_HEIGHT * (size_t)num_b + i] = v.z;
            state[SOIL_MOISTURE * (size_t)num_b + i] = -1.0f;
            state[GRID_POS * (size_t)num_b + i] = voxel_pos[i];
        }
        /* height above the local ground, for the wind profile: the ground of a raster cell is its ground height where
           the cell is open ground, elsewhere (under roofs, gaps) that of the nearest open ground cell (multi-source
           breadth-first search, 8-connected) */
        {
            std::vector<float> gz((size_t)W * H, 0.0f);
            std::vector<unsigned char> seen((size_t)W * H, 0);
            std::vector<int> queue;
            queue.reserve((size_t)W * H);
            for (int x = 0; x < W; ++x)
                for (int y = 0; y < H; ++y) {
                    const int c = sm.cell(x, y);
                    if (sm.z[TERRAIN_C][c] > 0.0f && sm.z[BUILDING_C][c] <= sm.z[TERRAIN_C][c]) {
                        const int k = x + y * W;
                        gz[k] = sm.z[TERRAIN_C][c]; seen[k] = 1; queue.push_back(k);
                    }
                }
            for (size_t q = 0; q < queue.size(); ++q) {
                const int k = queue[q], x = k % W, y = k / W;
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        const int n = nx + ny * W;
                        if (seen[n]) continue;
                        seen[n] = 1; gz[n] = gz[k]; queue.push_back(n);
                    }
            }
            for (unsigned int i = 0; i < num_b; ++i) {
                const int k = (int)voxel_pos[i];
                state[HEIGHT_AGL * (size_t)num_b + i] = (k >= 0 && k < W * H) ? std::max(sm.voxels[i].z - gz[k], 0.0f) : 0.0f;
            }
        }
        cudaSafeCall(cudaMemcpy(gridcells_data_d, state.data(), state.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    upload_model(mp); // model constants, material table, Perez coefficients

    /* packed height raster, in 1/SCALE_VAL m: vegetation in the lower 16 bits, the opaque surface (buildings or ground)
       in the upper; tree crowns are semi-transparent, so they are only in the vegetation height */
    {
        std::vector<unsigned int> tmp_grid((size_t)W * H);
        auto pack_z = [](float z) -> unsigned int {   // rounded to 1/SCALE_VAL m, clamped to 16 bits
            const long v = lroundf(z * SCALE_VAL);
            return (unsigned int)(v < 0 ? 0 : (v > 0xFFFF ? 0xFFFF : v));
        };
        for (int x = 0; x < W; ++x)
            for (int y = 0; y < H; ++y) {
                const int c = sm.cell(x, y);
                const float z_upper = (sm.z[BUILDING_C][c] > sm.z[TERRAIN_C][c]) ? sm.z[BUILDING_C][c] : sm.z[TERRAIN_C][c];
                tmp_grid[x + y * W] = (pack_z(z_upper) << 16) | pack_z(sm.z[VEGETATION_C][c]);
            }
        cudaSafeCall(cudaMemcpy(grid, tmp_grid.data(), tmp_grid.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    // ---------------- constants ----------------
    const float h_time_inter = 1.0f;
    const int tmp_num_hours = (int)(1.0f / h_time_inter) * 24;
    cudaSafeCall(cudaMemcpyToSymbol(width, (void*)&W, sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(height, (void*)&H, sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(maxz, (void*)&maxz_h, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(time_inter, (void*)&h_time_inter, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(num_data, (void*)&num_b, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(num_hours, (void*)&tmp_num_hours, sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(hres, (void*)&h_hres, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(vres, (void*)&h_vres, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(vres_f, (void*)&h_vres_f, sizeof(float), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(hsize, (void*)&h_hsize, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(bits_v, (void*)&hbits_v, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(vsize, (void*)&h_vsize, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(vsize_f, (void*)&h_vsize_f, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(gridRes, (void*)&h_gridRes, sizeof(float), 0, cudaMemcpyHostToDevice));

    // ---------------- hourly forcing ----------------
    upload_forcing(forcing, sun, tmp_num_hours);

    // ---------------- launch configuration ----------------
    dim3 dimBlock(CUDA_BLOCK_SIZE, 1, 1);
    dim3 dimGrid((int)(num_b / ((float)CUDA_BLOCK_SIZE)) + (num_b % (CUDA_BLOCK_SIZE) == 0 ? 0 : 1), 1, 1);

    cudaFuncSetCacheConfig(shortwave_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(viewshed_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(surroundings_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(wind_shelter_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(seb_kernel, cudaFuncCachePreferL1);

    // ---------------- 1. horizon buffers, sky view factor and viewshed caches ----------------
    unsigned int maxuniq = 0;
    cudaSafeCall(cudaMemcpyToSymbol(g_max_unique_in, (void*)&maxuniq, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    unsigned int max_indices = (360.0f / h_hres) * (180.0f / h_vres_f); // one slot per direction
    cudaSafeCall(cudaMemcpyToSymbol(max_indices_g, (void*)&max_indices, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    unsigned long long total_svf_hits = (unsigned long long)max_indices * (unsigned long long)num_b;
    std::cout << "Viewshed caches: " << num_b << " voxels x " << max_indices << " slots = "
              << total_svf_hits * sizeof(int) / (1024.0 * 1024.0 * 1024.0) << " GiB before compaction\n";
    cudaSafeCall(cudaMalloc((void**)&svf_ind, total_svf_hits * (unsigned long long)sizeof(int)));
    cudaSafeCall(cudaMemset(svf_ind, 0, total_svf_hits * (unsigned long long)sizeof(int)));
    print_gpu_memory("GPU memory");

    std::cout << "[1/2] Ray casting " << h_hsize * h_vsize_f << " directions: ";
    int step = (int)((float)(h_hsize * h_vsize_f) / 10.0f);
    for (int i = 0, n = 0; i < h_hsize; ++i) {
        for (int j = 0; j < h_vsize_f; ++j, ++n) {
            if (step > 0 && n != 0 && n % step == 0) std::cout << (int)(n / step) * 10 << "% " << std::flush;
            float3 Cdir;
            Cdir.x = cos(((j + 0.5f) * h_vres_f - 90.0f) * DTR) * sin(((i + 0.5f) * h_hres) * DTR);
            Cdir.y = -cos(((j + 0.5f) * h_vres_f - 90.0f) * DTR) * cos(((i + 0.5f) * h_hres) * DTR); // y grows southwards
            Cdir.z = sin(((j + 0.5f) * h_vres_f - 90.0f) * DTR);

            float len = sqrt(Cdir.x * Cdir.x + Cdir.y * Cdir.y + Cdir.z * Cdir.z);
            if (len > 0.0f) {
                Cdir.x /= len;
                Cdir.y /= len;
                Cdir.z /= len;
            }

            // solid angle of the direction bin
            float area_sphere = bin_solid_angle((j * h_vres_f - 90.0f) * DTR, (i * h_hres) * DTR,
                                                ((j + 1) * h_vres_f - 90.0f) * DTR, ((i + 1) * h_hres) * DTR);

            viewshed_kernel<<<dimGrid, dimBlock>>>(area_sphere, grid_s, grid_s_veg, grid_s_cover, grid_s_cover_veg, grid_s_front_hz, grid_w, Cdir, grid, gridcells_data_d, i, j, svf_ind, grid_t);
            cudaCheckError();
            cudaDeviceSynchronize();
        }
    }
    std::cout << "\n";

    // compact the viewshed caches to the slots in use
    unsigned int max_unique_indices = 0;
    cudaSafeCall(cudaMemcpyFromSymbol(&max_unique_indices, g_max_unique_in, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));
    {
        size_t new_n = static_cast<size_t>(num_b) * max_unique_indices;
        int* new_svf_ind = nullptr;
        cudaSafeCall(cudaMalloc(reinterpret_cast<void**>(&new_svf_ind), new_n * sizeof(int)));
        cudaSafeCall(cudaMemcpy(new_svf_ind, svf_ind, new_n * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaSafeCall(cudaFree(svf_ind));
        svf_ind = new_svf_ind;

        max_indices = max_unique_indices;
        cudaSafeCall(cudaMemcpyToSymbol(max_indices_g, (void*)&max_indices, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    }
    std::cout << "Viewshed caches: " << max_indices << " slots per voxel ("
              << (double)num_b * max_indices * sizeof(int) / (1024.0 * 1024.0 * 1024.0) << " GiB)\n";
    print_gpu_memory("GPU memory");

    // ---------------- 2. time stepping ----------------
    float* sun_circumsolar_region;
    std::vector<float> tmp_sun_circumsolar_region((size_t)24 * h_hsize * h_vsize, 0.0f);
    cudaSafeCall(cudaMalloc((void**)&sun_circumsolar_region, 24 * h_hsize * h_vsize * sizeof(float)));

    std::vector<float> state((size_t)NUM_ATTRIBUTES * num_b);

    const int start_h = rs.start_hour, end_h = rs.end_hour;
    const int total_hours = (rs.end_day - rs.start_day + rs.spinup_days) * (end_h - start_h);
    const int progress_interval = total_hours / 10;
    std::cout << "[2/2] Simulating " << (rs.end_day - rs.start_day) << " days";
    if (rs.spinup_days > 0) std::cout << " after " << rs.spinup_days << " spin-up days";
    std::cout << ": ";

    // spin-up days before 1 January wrap around to the end of the year (cyclic, e.g. a typical year)
    for (int nnn = 0, kd = rs.start_day - rs.spinup_days; kd < rs.end_day; ++kd) {
        const int n = (kd + 365) % 365;
        const bool spinup_day = kd < rs.start_day;

        // circumsolar region of every hour: the direction bins within CSA of the sun's bin, with their solid angles
        for (int h = start_h; h < end_h; ++h) {
            int ind = n * tmp_num_hours + h;

            circumsolar_region(std::get<0>(sun[ind]), std::get<1>(sun[ind]), h_hsize, h_vsize, h_hres, h_vres, rs.csa,
                               &tmp_sun_circumsolar_region[(size_t)h * h_vsize * h_hsize]);
        }
        cudaSafeCall(cudaMemcpy(sun_circumsolar_region, tmp_sun_circumsolar_region.data(), 24 * h_hsize * h_vsize * sizeof(float), cudaMemcpyHostToDevice));

        for (int h = start_h; h < end_h; ++h) {
            // direct + diffuse
            shortwave_kernel<<<dimGrid, dimBlock>>>(grid_s, grid_s_veg, grid_s_cover, grid_s_cover_veg, grid_s_front_hz, sun_circumsolar_region, gridcells_data_d, n, h);
            cudaCheckError();
            cudaDeviceSynchronize();

            // first-order reflection + longwave from the surroundings
            surroundings_kernel<<<dimGrid, dimBlock>>>(gridcells_data_d, n, h, svf_ind, 0);
            cudaCheckError();
            cudaDeviceSynchronize();
            // second-order reflection
            surroundings_kernel<<<dimGrid, dimBlock>>>(gridcells_data_d, n, h, svf_ind, 1);
            cudaCheckError();
            cudaDeviceSynchronize();

            wind_shelter_kernel<<<dimGrid, dimBlock>>>(grid_s, gridcells_data_d, n, h);
            cudaCheckError();
            cudaDeviceSynchronize();

            seb_kernel<<<dimGrid, dimBlock>>>(grid, grid_s_cover_veg, gridcells_data_d, n, h, start_h);
            cudaCheckError();
            cudaDeviceSynchronize();

            if (progress_interval > 0 && nnn % progress_interval == 0 && nnn != 0) {
                int percent = (int)(nnn / progress_interval) * 10;
                if (percent < 110) std::cout << percent << "% " << std::flush;
            }
            nnn++;

            if (!spinup_day) {
                cudaSafeCall(cudaMemcpy(state.data(), gridcells_data_d, state.size() * sizeof(float), cudaMemcpyDeviceToHost));
                on_hour(n, h, state);
            }
        }
    }
    std::cout << "\n";

    cudaSafeCall(cudaFree(sun_circumsolar_region));
    cudaSafeCall(cudaFree(grid));
    cudaSafeCall(cudaFree(grid_w));
    cudaSafeCall(cudaFree(grid_t));
    cudaSafeCall(cudaFree(grid_s));
    cudaSafeCall(cudaFree(grid_s_veg));
    cudaSafeCall(cudaFree(grid_s_cover));
    cudaSafeCall(cudaFree(grid_s_cover_veg));
    cudaSafeCall(cudaFree(grid_s_front_hz));
    cudaSafeCall(cudaFree(svf_ind));
    cudaSafeCall(cudaFree(gridcells_data_d));
    return 0;
}

} // namespace lstgpu
