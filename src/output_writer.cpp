#include "output_writer.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

namespace lstgpu {

OutputWriter::OutputWriter(const SurfaceModel& sm, const ModelParams&, const OutputSettings& os)
    : sm_(sm), os_(os), geo_(os.geo) {
    if (os.window) {
        out_x0_ = (int)lroundf((os.win_minx - sm.minx) / sm.res);
        out_y0_ = sm.height - (int)lroundf((os.win_maxy - sm.miny) / sm.res); // row 0 is the top of the raster
        out_w_ = (int)lroundf((os.win_maxx - os.win_minx) / sm.res);
        out_h_ = (int)lroundf((os.win_maxy - os.win_miny) / sm.res);
        geo_.origin_x = os.win_minx;
        geo_.origin_y = os.win_maxy;
    }
    else {
        out_w_ = sm.width;
        out_h_ = sm.height;
        geo_.origin_x = sm.minx;
        geo_.origin_y = (double)sm.miny + (double)sm.height * sm.res;
    }
    geo_.res = sm.res;
    std::filesystem::create_directories(os.dir);
}

template <class F> void OutputWriter::fill(int nbands, F value) {
    buf_.assign((size_t)out_w_ * out_h_ * nbands, std::numeric_limits<float>::quiet_NaN());
    for (int r = 0; r < out_h_; ++r) {
        const int y = out_y0_ + r;
        for (int c = 0; c < out_w_; ++c) {
            const int x = out_x0_ + c;
            if (x < 0 || x >= sm_.width || y < 0 || y >= sm_.height) continue;
            const int v = sm_.top_voxel[sm_.cell(x, y)];
            if (v == -1) continue;
            for (int b = 0; b < nbands; ++b) buf_[((size_t)r * out_w_ + c) * nbands + b] = value(v, b);
        }
    }
}

void OutputWriter::write_hour(int day, int hour, const std::vector<float>& state) {
    const size_t N = sm_.voxels.size();
    char name[32];
    std::snprintf(name, sizeof(name), "lst_d%03d_h%02d.tif", day, hour);
    fill(1, [&](int v, int) { return state[SURFACE_TEMPERATURE_H * N + v]; });
    write_tiff_bands((std::filesystem::path(os_.dir) / name).string(), out_w_, out_h_, buf_, { { "lst", "degC" } }, geo_);
}

} // namespace lstgpu
