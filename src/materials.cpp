#include "materials.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>

#include "geotiff_io.h"

namespace lstgpu {

int assign_materials(SurfaceModel& sm, const std::string& raster_path) {
    const int W = sm.width, H = sm.height;

    // from the surface classes of the height rasters
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) {
            const int v = sm.top_voxel[sm.cell(x, y)];
            if (v == -1) continue;
            float m = 0.0f;                                                // average ground
            if (sm.z[BUILDING_C][sm.cell(x, y)] > 0) m = 14.0f;            // average roof
            else if (sm.z[VEGETATION_C][sm.cell(x, y)] > 0) m = 5.0f;      // shrubs and trees
            sm.voxels[v].material = m;
        }
    if (raster_path.empty()) return 0;

    // material raster: a world file next to it gives the centre of the upper-left pixel, GeoTIFF tie points
    // (PixelIsArea) its corner
    float res_x = 0, res_y = 0, minx = 0, maxy = 0, tmp;
    std::ifstream tfw(std::regex_replace(raster_path, std::regex("(\\.tiff|\\.tif)$"), ".tfw"));
    const bool from_tfw = tfw.is_open();
    if (from_tfw) tfw >> res_x >> tmp >> tmp >> res_y >> minx >> maxy;

    TiffMeta meta;
    std::vector<float> classes = read_tiff_band(raster_path, meta);
    if (!from_tfw) {
        if (meta.pixelScale.size() < 2 || meta.tiepoints.size() < 5) {
            std::cerr << raster_path << ": no georeferencing (GeoTIFF tags or .tfw)\n";
            std::exit(EXIT_FAILURE);
        }
        res_x = (float)meta.pixelScale[0];
        res_y = (float)meta.pixelScale[1];
        minx = (float)meta.tiepoints[3];
        maxy = (float)meta.tiepoints[4];
    }

    const double rx = std::fabs(res_x), ry = std::fabs(res_y);
    const double ox = from_tfw ? minx - 0.5 * rx : minx; // left edge of the raster
    const double oy = from_tfw ? maxy + 0.5 * ry : maxy; // top edge
    int from_raster = 0;
    for (int i = 0; i < W; ++i)
        for (int j = 0; j < H; ++j) {
            const int v = sm.top_voxel[sm.cell(i, j)];
            if (v == -1) continue;
            // pixel under the cell centre; a centre exactly on a pixel edge takes the lower index
            const double xc = sm.minx + (i + 0.5) * sm.res;
            const double yc = sm.miny + ((double)H - j - 0.5) * sm.res;
            const int px = (int)std::floor((xc - ox) / rx - 1e-4);
            const int py = (int)std::floor((oy - yc) / ry - 1e-4);
            if (px < 0 || px >= (int)meta.w || py < 0 || py >= (int)meta.h) continue;
            const float value = classes[(size_t)py * meta.w + px];
            if (!(value >= 0.0f && value < (float)NUM_MATERIALS)) continue; // no data or not a class
            sm.voxels[v].material = (float)(int)value;
            ++from_raster;
        }
    std::cout << "Materials: " << from_raster << " of " << sm.num_top << " top voxels from " << raster_path << "\n";
    return from_raster;
}

} // namespace lstgpu
