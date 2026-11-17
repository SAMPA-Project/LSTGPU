#pragma once

/* Meteorological forcing: hourly series for one year (8760 values, one per line or whitespace separated; the
   value of hour h of day d is value number d * 24 + h) and a daily LAI series (365 values). The forcing folder
   holds one file per variable:

     direct.txt       direct (beam) irradiance on a horizontal plane (W m-2)
     diffuse.txt      diffuse irradiance on a horizontal plane (W m-2)
     temperature.txt  air temperature (degC)
     humidity.txt     relative humidity (%)
     windspeed.txt    wind speed at the reference height wind_zref (m s-1)
     winddir.txt      wind direction, where the wind comes from (degrees, clockwise from north)
     cloudiness.txt   total cloud cover (fraction 0..1)
     rainfall.txt     precipitation (mm per hour)
     lai.txt          leaf area index of the tree canopies, per day (m2 m-2)

   Longer series (e.g. 8784 values of a leap year) are cut to 365 days. */

#include <string>
#include <vector>

namespace lstgpu {

struct Forcing {
    std::vector<float> direct, diffuse, temperature, humidity, wind_speed, wind_dir, cloud, rain; // HOURS_PER_YEAR
    std::vector<float> lai;                                                                     // 365
};

/* Reads the forcing folder; `lai_path` overrides <dir>/lai.txt if not empty. Returns false and prints the
   reason if a file is missing or too short. */
bool read_forcing(const std::string& dir, const std::string& lai_path, Forcing& f);

} // namespace lstgpu
