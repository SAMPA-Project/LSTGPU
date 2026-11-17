#include "surface_raster.h"

#include <cmath>
#include <iostream>

namespace lstgpu {

namespace {

/* Slope (from horizontal) and aspect (clockwise from north) of a cell normal. */
void slope_aspect(float nx, float ny, float nz, float& slope, float& aspect) {
    float a = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (a > 0) a = std::acos(nz / a);
    if (a < 0.000001f || a > 1.3 || std::isnan(a) || std::isinf(a)) a = 0; // degenerate normals
    slope = a;
    a = std::atan2(nx, ny); // clockwise from north (+y)
    if (std::isnan(a)) a = PI;
    if (a < 0) a += 2.0f * PI;
    aspect = a;
}

/* Reads a single-band raster and checks it against the first one read. NaN becomes 0. */
bool read_band(const std::string& path, std::vector<float>& out, TiffMeta& meta, const TiffMeta* like) {
    out = read_tiff_band(path, meta);
    if (meta.w == 0 || meta.h == 0) {
        std::cerr << path << ": empty raster\n";
        return false;
    }
    if (meta.pixelScale.size() < 2 || meta.tiepoints.size() < 5) {
        std::cerr << path << ": no GeoTIFF georeferencing (ModelPixelScaleTag and ModelTiepointTag)\n";
        return false;
    }
    if (std::fabs(std::fabs(meta.pixelScale[0]) - std::fabs(meta.pixelScale[1])) > 1e-6) {
        std::cerr << path << ": the cells are not square (" << meta.pixelScale[0] << " x " << meta.pixelScale[1] << ")\n";
        return false;
    }
    if (like) {
        const bool same = meta.w == like->w && meta.h == like->h
                          && std::fabs(meta.pixelScale[0] - like->pixelScale[0]) < 1e-6
                          && std::fabs(meta.tiepoints[3] - like->tiepoints[3]) < 1e-6
                          && std::fabs(meta.tiepoints[4] - like->tiepoints[4]) < 1e-6;
        if (!same) {
            std::cerr << path << ": grid differs from the ground raster (" << meta.w << " x " << meta.h << " at "
                      << meta.tiepoints[3] << " " << meta.tiepoints[4] << " instead of " << like->w << " x " << like->h
                      << " at " << like->tiepoints[3] << " " << like->tiepoints[4] << ")\n";
            return false;
        }
    }
    for (auto& v : out)
        if (std::isnan(v) || std::isinf(v)) v = 0.0f;
    return true;
}

} // namespace

bool build_surface_model(const SurfaceRasters& in, SurfaceModel& sm, TiffMeta& crs) {
    std::vector<float> zg, zb, zv, nx, ny, nz;
    TiffMeta m;
    if (!read_band(in.ground, zg, crs, nullptr)) return false;
    if (!read_band(in.building, zb, m, &crs)) return false;
    if (!read_band(in.vegetation, zv, m, &crs)) return false;
    if (!read_band(in.normal_x, nx, m, &crs)) return false;
    if (!read_band(in.normal_y, ny, m, &crs)) return false;
    if (!read_band(in.normal_z, nz, m, &crs)) return false;

    const int W = sm.width = (int)crs.w;
    const int H = sm.height = (int)crs.h;
    sm.res = (float)std::fabs(crs.pixelScale[0]);
    sm.minx = (float)crs.tiepoints[3];                            // left edge (the tie point is PixelIsArea)
    sm.miny = (float)(crs.tiepoints[4] - (double)H * sm.res);     // bottom edge
    for (auto& zc : sm.z) zc.assign((size_t)W * H, 0.0f);

    /* The rasters are row-major from the north edge, the surface model is x-major with row 0 at the north edge, so
       raster pixel (x, y) is cell(x, y). */
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) {
            const size_t k = (size_t)y * W + x;
            const int c = sm.cell(x, y);
            sm.z[TERRAIN_C][c] = zg[k] > 0.0f ? zg[k] : 0.0f;
            sm.z[BUILDING_C][c] = zb[k] > 0.0f ? zb[k] : 0.0f;
            sm.z[VEGETATION_C][c] = zv[k] > 0.0f ? zv[k] : 0.0f;
        }

    // ---- no vegetation over buildings (as in the point-cloud surface model) ----
    for (int c = 0; c < W * H; ++c)
        if (sm.z[BUILDING_C][c] != 0) sm.z[VEGETATION_C][c] = 0;

    // ---- top surface: building, else vegetation, else ground ----
    for (int c = 0; c < W * H; ++c) {
        if (sm.z[BUILDING_C][c] > sm.z[TERRAIN_C][c]) sm.z[OBJECT_C][c] = sm.z[BUILDING_C][c];
        else if (sm.z[VEGETATION_C][c] != 0) sm.z[OBJECT_C][c] = sm.z[VEGETATION_C][c];
        else if (sm.z[TERRAIN_C][c] != 0) sm.z[OBJECT_C][c] = sm.z[TERRAIN_C][c];
    }

    // ---- heights relative to the lowest ground cell ----
    float minz = 9999999;
    for (int c = 0; c < W * H; ++c) {
        const float zt = sm.z[TERRAIN_C][c];
        if (zt != 0 && zt > 0.001f && zt < minz) minz = zt;
    }
    if (minz > 9999998) {
        std::cerr << in.ground << ": no ground cell with a height\n";
        return false;
    }
    /* the reference on the absolute 1/SCALE_VAL m lattice and one step below the lowest ground cell, so that the
       packed height raster rounds the same absolute heights whatever the extent of the simulated area; the lowest
       cell stays above 0 (= no data) */
    minz = std::floor(minz * SCALE_VAL) / SCALE_VAL - 1.0f / SCALE_VAL;
    sm.z_offset = minz;

    // ---- top voxels, x-major raster order ----
    sm.top_voxel.assign((size_t)W * H, -1);
    sm.extra_voxels.assign((size_t)W * H, std::vector<int>());
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) {
            const int c = sm.cell(x, y);
            for (int k : { TERRAIN_C, BUILDING_C, OBJECT_C, VEGETATION_C })
                if (sm.z[k][c] != 0) sm.z[k][c] -= minz;
            if (sm.z[OBJECT_C][c] == 0) continue;

            Voxel v;
            v.x = x;
            v.y = y;
            v.z = sm.z[OBJECT_C][c];
            v.material = (sm.z[BUILDING_C][c] > sm.z[TERRAIN_C][c]) ? 6.0f : (sm.z[VEGETATION_C][c] > sm.z[TERRAIN_C][c]) ? 5.0f : 2.0f;
            sm.top_voxel[c] = (int)sm.voxels.size();
            sm.voxels.push_back(v);
        }
    sm.num_top = (int)sm.voxels.size();

    // ---- slope and aspect of the top voxels from the cell normals ----
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) {
            const int v = sm.top_voxel[sm.cell(x, y)];
            if (v == -1) continue;
            const size_t k = (size_t)y * W + x;
            slope_aspect(nx[k], ny[k], nz[k], sm.voxels[v].slope, sm.voxels[v].aspect);
            if (sm.voxels[v].slope < 0.0174533f) {
                sm.voxels[v].slope = 0;
                sm.voxels[v].aspect = PI;
            }
        }

    // ---- ground under tree canopies ----
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) {
            const int c = sm.cell(x, y);
            if (!(sm.z[VEGETATION_C][c] > sm.z[TERRAIN_C][c])) continue;
            Voxel v;
            v.x = x;
            v.y = y;
            v.z = sm.z[TERRAIN_C][c];
            v.slope = 0.0001f;
            v.aspect = PI;
            v.material = 2.0f; // soil
            v.under_canopy = true;
            sm.extra_voxels[c].push_back((int)sm.voxels.size());
            sm.voxels.push_back(v);
        }

    // ---- wall slices (1 m) on building edges higher than 1.1 m above an axial neighbour ----
    const float WALL_HEIGHT_THRESHOLD = 1.1f;
    const float WALL_CELL_HEIGHT = 1.0f;
    const int axial[4][2] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } }; // the order of the original 3x3 scan
    auto zbc = [&](int x, int y) { return sm.z[BUILDING_C][sm.cell(x, y)]; };
    auto ztc = [&](int x, int y) { return sm.z[TERRAIN_C][sm.cell(x, y)]; };
    // neighbour surface height; outside the raster: the cell's own ground (open)
    auto neighbour_height = [&](int x, int y, int nbx, int nby) {
        if (nbx >= 0 && nbx < W && nby >= 0 && nby < H) return std::max(zbc(nbx, nby), ztc(nbx, nby));
        return ztc(x, y);
    };

    auto add_walls = [&](int x, int y) {
        for (const auto& d : axial) {
            if (!((zbc(x, y) - neighbour_height(x, y, x + d[0], y + d[1])) > WALL_HEIGHT_THRESHOLD)) continue;

            int open_slices = 0;
            for (float zz = zbc(x, y) - WALL_CELL_HEIGHT; zz >= ztc(x, y) - WALL_CELL_HEIGHT; zz -= WALL_CELL_HEIGHT) {
                bool exposed = false;
                for (const auto& d2 : axial)
                    if ((zz - neighbour_height(x, y, x + d2[0], y + d2[1])) > WALL_HEIGHT_THRESHOLD) exposed = true;
                if (!exposed) open_slices++;
                if (open_slices == 2) return; // the wall ends a slice below the neighbours' height

                // aspect: towards the more open side (0 = north = -y, 90 = east = +x)
                const float OE = std::fmax(0.0f, zz - neighbour_height(x, y, x + 1, y));
                const float OW = std::fmax(0.0f, zz - neighbour_height(x, y, x - 1, y));
                const float ON = std::fmax(0.0f, zz - neighbour_height(x, y, x, y - 1));
                const float OS = std::fmax(0.0f, zz - neighbour_height(x, y, x, y + 1));
                float aspect = std::atan2(OE - OW, ON - OS);
                if (aspect < 0.0f) aspect += 2.0f * PI;

                Voxel v;
                v.x = x;
                v.y = y;
                v.z = zz;
                v.slope = PI * 0.499f;
                v.aspect = aspect;
                v.material = 11.0f; // wall
                v.wall = true;
                sm.extra_voxels[sm.cell(x, y)].push_back((int)sm.voxels.size());
                sm.voxels.push_back(v);
            }
            return;
        }
    };
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y)
            if (zbc(x, y) > ztc(x, y)) add_walls(x, y);

    std::cout << "Surface model: " << W << " x " << H << " cells of " << sm.res << " m, " << sm.num_top
              << " top voxels, " << (sm.voxels.size() - sm.num_top) << " wall and under-canopy voxels, height "
              << "reference " << sm.z_offset << " m\n";
    return true;
}

} // namespace lstgpu
