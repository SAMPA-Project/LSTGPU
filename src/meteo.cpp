#include "meteo.h"

#include <fstream>
#include <iostream>

#include "lst_constants.h"

namespace lstgpu {

namespace {

bool read_series(const std::string& path, size_t n, std::vector<float>& v) {
    std::ifstream ifd(path);
    if (!ifd.is_open()) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }
    v.clear();
    v.reserve(n);
    float x;
    while (v.size() < n && (ifd >> x)) v.push_back(x);
    if (v.size() < n) {
        std::cerr << path << ": " << v.size() << " values, " << n << " needed\n";
        return false;
    }
    return true;
}

// Irradiance outside 0..1367 W m-2 (sensor errors): negative -> 0, too high -> `fallback`
void check_irradiance(std::vector<float>& v, float fallback, const char* name) {
    int fixed = 0;
    for (auto& x : v) {
        if (x < 0) { x = 0; ++fixed; }
        if (x > 1367) { x = fallback; ++fixed; }
    }
    if (fixed) std::cout << name << ": " << fixed << " values outside 0..1367 W m-2 replaced\n";
}

} // namespace

bool read_forcing(const std::string& dir, const std::string& lai_path, Forcing& f) {
    const std::string d = dir + "/";
    const size_t n = HOURS_PER_YEAR;
    bool ok = read_series(d + "direct.txt", n, f.direct)
        && read_series(d + "diffuse.txt", n, f.diffuse)
        && read_series(d + "temperature.txt", n, f.temperature)
        && read_series(d + "humidity.txt", n, f.humidity)
        && read_series(d + "windspeed.txt", n, f.wind_speed)
        && read_series(d + "winddir.txt", n, f.wind_dir)
        && read_series(d + "cloudiness.txt", n, f.cloud)
        && read_series(d + "rainfall.txt", n, f.rain)
        && read_series(lai_path.empty() ? d + "lai.txt" : lai_path, 365, f.lai);
    if (!ok) return false;
    check_irradiance(f.diffuse, 341.75f, "diffuse");
    check_irradiance(f.direct, 683.5f, "direct");
    return true;
}

} // namespace lstgpu
