#pragma once

/* Uploads the model and the forcing to the device globals of device_globals.cuh. */

#include <vector>

#include "meteo.h"
#include "model_params.h"
#include "sun_position.h"

namespace lstgpu {

// Model constants (P_g), material table (class_matrix) and Perez coefficients (F_matrix)
void upload_model(const ModelParams& mp);

/* Extraterrestrial irradiance per day (E0) and the hourly forcing of the year, num_hours per day: sun position,
   direct and diffuse irradiance, air temperature, vapour pressure (from the relative humidity), wind, cloud cover,
   rain, and the daily LAI. */
void upload_forcing(const Forcing& forcing, const std::vector<SunSample>& sun, int num_hours);

} // namespace lstgpu
