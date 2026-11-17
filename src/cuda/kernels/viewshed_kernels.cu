#include "cuda/kernels/viewshed_kernels.cuh"

#include "cuda/physics/surface_geometry.cuh"

#define RAY_EPS_SCALAR  0.01f // ~0.001 of a cell
#define RAY_EPS_XY      0.01f // tiny along-ray XY nudge

__global__ void viewshed_kernel(float patch_area, grid_s_type* grid_s, grid_s_type* grid_s_veg, float2* grid_s_cover,
    float2* grid_s_cover_veg, float* grid_s_front_hz, const unsigned int* grid_w, float3 Cdir, const unsigned int* grid,
    const float* targetData, int i, int j, int* svf_ind, const unsigned int* grid_t) {
    const unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id >= num_data) return;

    float3 Ci_tid, O;

    unsigned int g = targetData[(NUM_ATTRIBUTES - 1) * num_data + t_id];
    float slope = targetData[SLOPE * num_data + t_id];
    float aspect = targetData[ASPECT * num_data + t_id];
    int init_x, init_y;
    init_x = Ci_tid.x = g % width;
    init_y = Ci_tid.y = ((float)g / (float)width);

    Ci_tid.x = (float)init_x + 0.5f;
    Ci_tid.y = (float)init_y + 0.5f;
    Ci_tid.z = fmaxf(targetData[CELL_HEIGHT * num_data + t_id], 0.1f);

    const float3 nrm = surface_normal(slope, aspect);
    float angle_of_incidence = nrm.x * Cdir.x + nrm.y * Cdir.y + nrm.z * Cdir.z;

    int start_j = vsize_f / 2.0f; // first bin of the upper hemisphere

    const bool sky_patch = (j - start_j >= 0);
    const bool hz_patch = sky_patch && (j - start_j == 0 || (j - start_j + 1) * vres < 5);
    const float w_cos = patch_area * fmaxf(angle_of_incidence, 0.0f); // cosine-weighted solid angle, 0 behind the surface
    if (hz_patch) grid_s_front_hz[t_id] += w_cos;

    if (angle_of_incidence < 0.0f) { // direction behind the surface: self-shadowing of the direct beam
        if (sky_patch) horizon_raise(t_id, (grid_s_type)i, (grid_s_type)(j - start_j), grid_s);
    }

    int src_id = 0;
    float t = 0.0f; // ray parameter

    // tiny nudge to avoid starting exactly on a face (reduces self-shadowing)
    const float eps_n = RAY_EPS_SCALAR * gridRes; // lift off the surface
    const float eps_xy = RAY_EPS_XY;              // XY nudge in grid units

    O.x = Ci_tid.x + eps_xy * Cdir.x + eps_n * nrm.x;
    O.y = Ci_tid.y + eps_xy * Cdir.y + eps_n * nrm.y;
    O.z = Ci_tid.z + eps_n * nrm.z;

    // ---- DDA setup ----
    int cellX = (int)floorf(O.x);
    int cellY = (int)floorf(O.y);

    int stepX = (Cdir.x > 0.0f) ? 1 : (Cdir.x < 0.0f ? -1 : 0);
    int stepY = (Cdir.y > 0.0f) ? 1 : (Cdir.y < 0.0f ? -1 : 0);

    float invDx = (Cdir.x != 0.0f) ? 1.0f / Cdir.x : INFINITY;
    float invDy = (Cdir.y != 0.0f) ? 1.0f / Cdir.y : INFINITY;

    // next grid lines in the travel direction
    float nextVLine = (stepX > 0) ? ((float)cellX + 1.0f) : ((float)cellX);
    float nextHLine = (stepY > 0) ? ((float)cellY + 1.0f) : ((float)cellY);

    // ray parameter at those lines, and per cell crossed
    float tMaxX = (stepX != 0) ? (nextVLine - O.x) * invDx : INFINITY;
    float tMaxY = (stepY != 0) ? (nextHLine - O.y) * invDy : INFINITY;
    float tDeltaX = (stepX != 0) ? fabsf(invDx) : INFINITY;
    float tDeltaY = (stepY != 0) ? fabsf(invDy) : INFINITY;

    int s_veg = 0; // vegetation cells passed
    bool escaped = false;

    while (1) {
        if (tMaxX < tMaxY) {
            t = tMaxX;
            tMaxX += tDeltaX;
            cellX += stepX;
        }
        else {
            t = tMaxY;
            tMaxY += tDeltaY;
            cellY += stepY;
        }

        Ci_tid.z = O.z + t * Cdir.z;

        if (cellX < 0 || cellX >= width || cellY < 0 || cellY >= height || Ci_tid.z > maxz) { escaped = true; break; } // sky

        src_id = cellX + cellY * width;

        if ((cellX == init_x && cellY == init_y)) continue;

        if ((GRID_Z(grid[src_id]) - Ci_tid.z > 0.5)) { // hit terrain or a building
            int is_wall = 0;

            int wall_cells_ind_size = grid_w[2 * (src_id) + 0];

            if (wall_cells_ind_size > 0 && GRID_Z(grid[src_id]) - Ci_tid.z > gridRes) { // below the roof: a wall voxel
                int offset = grid_w[2 * (src_id) + 1];

                for (int w = wall_cells_ind_size - 1; w >= 0; w--) {
                    float wall_z = targetData[CELL_HEIGHT * num_data + (offset + w)];

                    if (wall_z - Ci_tid.z > 0.01) {
                        src_id = (offset + w);
                        is_wall = 1;
                        break;
                    }
                }
            }

            if (!is_wall && (GRID_Z(grid[src_id]) - Ci_tid.z > 1.1)) { // the ray went inside a building
                return;
            }

            /* svf_ind holds voxel indices: the wall branch above already stored one (offset + w, a row of targetData),
               so a top cell has to be converted from its raster index too. The two are not the same numbering, because
               voxels are handed out over the occupied cells only and in x-major order, so a raster index addresses
               another cell's voxel, or none at all where the raster holds more cells than there are voxels. */
            if (!is_wall) {
                const unsigned int tgt = grid_t[src_id];
                if (tgt == GRID_NO_TARGET) return; // no surface in this cell
                src_id = (int)tgt;
            }

            // horizon buffer together with the cache (the obstructed patch adds nothing to the sky view)
            if (j - start_j >= 0 && angle_of_incidence >= 0.0001f) {
                horizon_raise(t_id, (grid_s_type)i, (grid_s_type)(j - start_j), grid_s);
            }

            unsigned int id;
            for (unsigned int ii = 0; ii < max_indices_g; ii++) {
                id = svf_ind[ii * num_data + t_id];
                if (id == 0) { // free slot
                    if (s_veg > 0) src_id = -src_id;
                    svf_ind[ii * num_data + t_id] = src_id;
                    atomicMax(&g_max_unique_in, ii);
                    return;
                }
                else if (P_g[P_LSURR_RAYW] > 0.0f) { // voxel already hit by another ray: count the ray (each ray stands for its direction bin)
                    int e = (int)id, m = abs(e);
                    if ((m & SVF_ID_MASK) == src_id) {
                        if ((m >> SVF_CNT_SHIFT) < 255) m += 1 << SVF_CNT_SHIFT;
                        svf_ind[ii * num_data + t_id] = (e < 0) ? -m : m;
                        return;
                    }
                }
                else if (id == src_id) { // duplicate (the directions are sampled per solid-angle bin)
                    return;
                }
            }

            break;
        }
        else if ((GRID_ZV(grid[src_id]) - Ci_tid.z > 0.01)) { // vegetation
            s_veg++;

            if (j - start_j > 0 && s_veg == 1) {
                horizon_raise(t_id, (grid_s_type)i, (grid_s_type)(j - start_j), grid_s_veg);
            }
        }
    }

    // sky view: the cosine-weighted solid angle of a patch whose ray reaches the sky, and the part of it seen through the
    // canopy (written after the ray loop: stores inside it slowed the kernel down ~4x)
    if (escaped && sky_patch && w_cos > 0.0f) {
        float2 c = grid_s_cover[t_id];
        c.x += w_cos;
        if (hz_patch) c.y += w_cos;
        grid_s_cover[t_id] = c;
        if (s_veg > 0) {
            float2 v = grid_s_cover_veg[t_id];
            v.x += w_cos;
            if (hz_patch) v.y += w_cos;
            grid_s_cover_veg[t_id] = v;
        }
    }
}
