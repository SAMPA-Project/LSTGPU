#pragma once

/* 2.5D surface model of the simulated area and its surface voxels, read from rasters.

   The area is given as single-band GeoTIFF rasters of the same grid: the ground, building and canopy heights
   (absolute elevation in map units, 0 or NaN where the class is absent) and the three components of the surface
   normal of every cell. Every cell with a surface becomes a "top" voxel with a slope and aspect from its normal.
   Two kinds of extra voxels are stacked under the top voxels: the ground under a tree canopy, and 1 m wall slices
   on building edges. All heights are stored relative to a reference derived from the lowest ground cell. */

#include <string>
#include <vector>

#include "geotiff_io.h"
#include "lst_constants.h"

namespace lstgpu {

struct Voxel {
    int x = 0, y = 0;                  // raster cell of the voxel's column (row 0 is the top / north edge)
    float z = 0.0f;                    // height above the height reference (m)
    float slope = 0.0f, aspect = 0.0f; // radians; aspect clockwise from north
    float material = 0.0f;             // row of the material table
    bool under_canopy = false;         // ground voxel under a tree canopy
    bool wall = false;                 // wall slice
};

struct SurfaceModel {
    int width = 0, height = 0;
    float res = 1.0f;
    float minx = 0.0f, miny = 0.0f; // lower-left corner of the raster; row 0 starts at miny + height * res
    float z_offset = 0.0f;          // subtracted from all heights: lowest ground cell, rounded down to
                                    // 1/SCALE_VAL m, - 1/SCALE_VAL m

    std::vector<float> z[NUM_CLASSES];          // per class, indexed x * height + y
    std::vector<int> top_voxel;                 // per raster cell, -1 = no surface
    std::vector<std::vector<int>> extra_voxels; // per raster cell, under-canopy and wall voxels of the column

    /* Top voxels first, in x-major raster order (x outer, y inner), then all under-canopy voxels, then all wall
       voxels. */
    std::vector<Voxel> voxels;
    int num_top = 0;

    int cell(int x, int y) const { return x * height + y; }
};

/* The six input rasters of the surface. */
struct SurfaceRasters {
    std::string ground, building, vegetation; // absolute elevation (m), 0 or NaN = the class is absent in the cell
    std::string normal_x, normal_y, normal_z; // surface normal of the cell, z upwards; all three 0 = flat
};

/* Builds the surface model from `in`. All six rasters must have the same size, square cells and placement. On
   success the coordinate reference system of the ground raster is returned in `crs`. Returns false and prints the
   reason if a raster can't be used. */
bool build_surface_model(const SurfaceRasters& in, SurfaceModel& sm, TiffMeta& crs);

} // namespace lstgpu
