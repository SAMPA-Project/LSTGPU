#pragma once

/* Writes the results of the top voxels as GeoTIFF rasters (the wall and under-canopy voxels take part in the
   simulation but have no place in a top-view raster):
     <out>/lst_dDDD_hHH.tif              land surface temperature (degC) of every simulated hour
   DDD is the 0-based day of the year, HH the hour. Cells without a surface are NaN. */

#include <string>
#include <vector>

#include "geotiff_io.h"
#include "model_params.h"
#include "surface_raster.h"

namespace lstgpu {

struct OutputSettings {
    std::string dir;
    bool window = false;                                    // write only this map window (e.g. a tile without its buffer)
    float win_minx = 0, win_miny = 0, win_maxx = 0, win_maxy = 0;
    GeoRef geo;                                             // CRS keys; the placement is set by the writer
};

class OutputWriter {
public:
    OutputWriter(const SurfaceModel& sm, const ModelParams& mp, const OutputSettings& os);

    void write_hour(int day, int hour, const std::vector<float>& state);

    int width() const { return out_w_; }
    int height() const { return out_h_; }
    const GeoRef& georef() const { return geo_; }

private:
    const SurfaceModel& sm_;
    OutputSettings os_;
    GeoRef geo_;
    int out_x0_ = 0, out_y0_ = 0, out_w_ = 0, out_h_ = 0;
    std::vector<float> buf_;

    // fills buf_ with value(voxel, band) for the top voxel of every output pixel, NaN elsewhere
    template <class F> void fill(int nbands, F value);
};

} // namespace lstgpu
