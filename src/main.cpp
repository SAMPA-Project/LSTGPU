/* LSTGPU command line: given raster surface data and hourly meteorological data, simulates the land
   surface temperature and the surface energy balance of every 1 m surface voxel on the GPU and writes them as
   hourly GeoTIFF rasters. */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "geotiff_io.h"
#include "lst_gpu_api.h"
#include "materials.h"
#include "meteo.h"
#include "model_params.h"
#include "output_writer.h"
#include "sun_position.h"
#include "surface_raster.h"

namespace fs = std::filesystem;
using namespace lstgpu;

namespace {

const char* kVersion = "1.0";

const char* kUsage =
    "LSTGPU " "1.0" " - GPU-accelerated land surface temperature simulation over raster surface data\n"
    "\n"
    "usage: LSTGPU --surface <prefix> --meteo <dir> --out <dir> --lon <deg> --lat <deg> --timezone <h> [options]\n"
    "\n"
    "required:\n"
    "  --surface <prefix>    six single-band GeoTIFF rasters of one grid, <prefix>_ground.tif _building.tif\n"
    "                        _vegetation.tif (absolute surface elevation in m, 0 where the class is absent in the\n"
    "                        cell) and <prefix>_normal_x.tif _normal_y.tif _normal_z.tif (surface normal, z upwards);\n"
    "                        a folder instead of a prefix reads ground.tif ... normal_z.tif in it\n"
    "  --meteo <dir>         hourly meteorological data (direct.txt diffuse.txt temperature.txt humidity.txt\n"
    "                        windspeed.txt winddir.txt cloudiness.txt rainfall.txt, 8760 values each) and\n"
    "                        lai.txt (365 values)\n"
    "  --out <dir>           output folder (created)\n"
    "  --lon, --lat <deg>    geographic coordinates of the area (for the sun position)\n"
    "  --timezone <h>        UTC offset of the data's clock (fixed, no daylight saving time), e.g. 1 for CET\n"
    "\n"
    "options:\n"
    "  --days <a> <b>        simulate days a .. b-1 (0 = 1 January; default 0 365)\n"
    "  --spinup <n>          simulate n days before day a without writing them (default 0); before 1 January they\n"
    "                        wrap around to the end of the year, which suits a typical meteorological year\n"
    "  --hours <a> <b>       simulate hours a .. b-1 of every day (default 0 24)\n"
    "  --year <y>            calendar year of the data for the sun position; a leap year has 29 February in its\n"
    "                        series (default 2017, i.e. a non-leap typical year)\n"
    "  --materials <tif>     material class raster (0..14, the material table of the model constants); default: from the height rasters\n"
    "  --lai <file>          daily LAI instead of <meteo>/lai.txt\n"
    "  --params <file>       model constants (key value lines); default: the calibrated values\n"
    "  --window <minx> <miny> <maxx> <maxy>\n"
    "                        write only this map window, e.g. the study area without its boundary\n"
    "  --epsg <code>         CRS of the outputs (default: from the ground raster)\n"
    "  --hres, --vres <deg>  hemisphere bin size, azimuth and elevation (default 5 5)\n"
    "  --csa <deg>           circumsolar region half-angle (default 1)\n";

struct Args {
    std::map<std::string, std::vector<std::string>> a;
    bool has(const std::string& k) const { return a.count(k) > 0; }
    std::string str(const std::string& k, const std::string& d = "", size_t i = 0) const {
        auto it = a.find(k);
        return (it != a.end() && it->second.size() > i) ? it->second[i] : d;
    }
    float num(const std::string& k, float d, size_t i = 0) const {
        auto it = a.find(k);
        if (it == a.end() || it->second.size() <= i) return d;
        try {
            return std::stof(it->second[i]);
        }
        catch (...) {
            std::cerr << "Not a number: " << k << " " << it->second[i] << "\n";
            std::exit(2);
        }
    }
};

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc;) {
        std::string k = argv[i++];
        std::vector<std::string> v;
        while (i < argc && !(argv[i][0] == '-' && argv[i][1] == '-')) v.push_back(argv[i++]);
        args.a[k] = v;
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    const Args args = parse(argc, argv);
    if (argc < 2 || args.has("--help") || args.has("-h")) {
        std::cout << kUsage;
        return argc < 2 ? 2 : 0;
    }

    static const char* known[] = { "--surface", "--meteo", "--out", "--lon", "--lat", "--timezone", "--days",
        "--spinup", "--hours", "--year", "--materials", "--lai", "--params", "--window", "--epsg",
        "--hres", "--vres", "--csa" };
    for (const auto& kv : args.a) {
        bool ok = false;
        for (const char* k : known) ok = ok || kv.first == k;
        if (!ok) {
            std::cerr << "Unknown option " << kv.first << "\n\n" << kUsage;
            return 2;
        }
    }
    for (const char* k : { "--surface", "--meteo", "--out", "--lon", "--lat", "--timezone" })
        if (args.str(k).empty()) {
            std::cerr << "Missing " << k << "\n\n" << kUsage;
            return 2;
        }

    RunSettings rs;
    rs.start_day = (int)args.num("--days", 0, 0);
    rs.end_day = (int)args.num("--days", 365, 1);
    rs.spinup_days = (int)args.num("--spinup", 0);
    rs.start_hour = (int)args.num("--hours", 0, 0);
    rs.end_hour = (int)args.num("--hours", 24, 1);
    rs.hres = args.num("--hres", 5);
    rs.vres = args.num("--vres", 5);
    rs.csa = args.num("--csa", 1);
    if (rs.start_day < 0 || rs.end_day > 365 || rs.start_day >= rs.end_day || rs.spinup_days < 0 || rs.spinup_days > 365
        || rs.start_hour < 0 || rs.end_hour > 24 || rs.start_hour >= rs.end_hour || rs.hres <= 0 || rs.vres <= 0) {
        std::cerr << "Invalid --days, --spinup, --hours, --hres or --vres\n";
        return 2;
    }
    const float lon = args.num("--lon", 0), lat = args.num("--lat", 0), timezone = args.num("--timezone", 0);
    const int year = (int)args.num("--year", 2017);
    const std::string out_dir = args.str("--out");

    register_geotiff_tags();

    // ---- model constants ----
    ModelParams mp = default_params();
    if (args.has("--params")) {
        std::string log;
        if (!read_params(args.str("--params"), mp, log)) {
            std::cerr << log;
            return 1;
        }
        std::cout << log;
    }

    // ---- surface model from the rasters ----
    const std::string surf = args.str("--surface");
    // <prefix>_ground.tif ..., or ground.tif ... in a folder
    auto surface_file = [&](const char* layer) {
        return fs::is_directory(surf) ? (fs::path(surf) / (std::string(layer) + ".tif")).string()
                                      : surf + "_" + layer + ".tif";
    };
    SurfaceRasters sr;
    sr.ground = surface_file("ground");
    sr.building = surface_file("building");
    sr.vegetation = surface_file("vegetation");
    sr.normal_x = surface_file("normal_x");
    sr.normal_y = surface_file("normal_y");
    sr.normal_z = surface_file("normal_z");
    for (const std::string* f : { &sr.ground, &sr.building, &sr.vegetation, &sr.normal_x, &sr.normal_y, &sr.normal_z })
        if (!fs::exists(*f)) {
            std::cerr << "Missing surface raster " << *f << "\n";
            return 1;
        }
    SurfaceModel sm;
    TiffMeta crs;
    if (!build_surface_model(sr, sm, crs)) return 1;

    assign_materials(sm, args.str("--materials"));

    // ---- meteorological data and sun ----
    Forcing forcing;
    if (!read_forcing(args.str("--meteo"), args.str("--lai"), forcing)) return 1;
    std::vector<SunSample> sun;
    SunPositionCalculator(lon, lat, timezone, year).calc(sun);

    fs::create_directories(out_dir);

    // ---- outputs ----
    OutputSettings os;
    os.dir = out_dir;
    os.window = args.has("--window");
    if (os.window) {
        os.win_minx = args.num("--window", 0, 0);
        os.win_miny = args.num("--window", 0, 1);
        os.win_maxx = args.num("--window", 0, 2);
        os.win_maxy = args.num("--window", 0, 3);
        if (!(os.win_maxx > os.win_minx && os.win_maxy > os.win_miny)) {
            std::cerr << "Invalid --window\n";
            return 2;
        }
    }
    const int epsg = (int)args.num("--epsg", 0);
    if (args.has("--epsg")) set_crs(os.geo, {}, {}, "", epsg);
    else set_crs(os.geo, crs.geoKeys, crs.geoDoubles, crs.geoAscii, 0);
    if (os.geo.geoKeys.empty()) std::cout << "No CRS in the ground raster and no --epsg: the outputs have no CRS\n";
    OutputWriter writer(sm, mp, os);

    // ---- simulation ----
    int hours_written = 0;
    const int rc = run_lst_gpu(sm, forcing, sun, mp, rs, [&](int day, int hour, const std::vector<float>& state) {
        writer.write_hour(day, hour, state);
        ++hours_written;
    });
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- run summary ----
    std::cout << "LSTGPU " << kVersion << ", status " << (rc == 0 ? "ok" : "failed") << "\n"
              << "output rasters: " << writer.width() << " x " << writer.height() << ", upper-left corner " << std::fixed
              << writer.georef().origin_x << " " << writer.georef().origin_y << "\n"
              << "days: " << rs.start_day << " - " << rs.end_day - 1 << " (spin-up " << rs.spinup_days << "), hours: "
              << rs.start_hour << " - " << rs.end_hour - 1 << "\n"
              << "Done in " << secs << " s, " << hours_written << " LST rasters in " << out_dir << "\n";
    return rc;
}
