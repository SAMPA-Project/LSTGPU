#pragma once

/* Host-facing entry point into the CUDA pipeline. This header has no CUDA dependency so it can be included from
   plain C++ translation units like src/main.cpp. */

#include <functional>
#include <vector>

#include "meteo.h"
#include "model_params.h"
#include "surface_raster.h"
#include "sun_position.h"

namespace lstgpu {

struct RunSettings {
    int start_day = 0, end_day = 365;  // simulated days [start_day, end_day), 0 = 1 January
    int spinup_days = 0;               // days simulated before start_day and not reported; before 1 January they
                                       // wrap around to the end of the year (cyclic, e.g. a typical year)
    int start_hour = 0, end_hour = 24; // simulated hours of every day [start_hour, end_hour)
    float hres = 5.0f, vres = 5.0f;    // hemisphere bin size (degrees): azimuth, elevation
    float csa = 1.0f;                  // circumsolar region half-angle (degrees)
};

/* Called after every simulated hour outside the spin-up, with the GPU state of all voxels
   (NUM_ATTRIBUTES x voxels, attribute-major, see lst_constants.h). `day` is 0-based. */
using HourCallback = std::function<void(int day, int hour, const std::vector<float>& state)>;

/* Runs the simulation for the voxels of `sm`. `sun` holds (azimuth, altitude) in degrees for every hour of the
   year. Returns 0 on success. */
int run_lst_gpu(const SurfaceModel& sm, const Forcing& forcing, const std::vector<SunSample>& sun,
                const ModelParams& mp, const RunSettings& rs, const HourCallback& on_hour);

} // namespace lstgpu
