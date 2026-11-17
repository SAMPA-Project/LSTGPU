#include "model_params.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace lstgpu {

const char* const kParamNames[P_NUM] = {
    "hc_scale", "hc_min", "wind_min", "wind_exp", "wind_zref", "seasonal_storage_amp", "seasonal_storage_phase_day",
    "canopy_k", "wind_canopy_k", "wind_canopy_red", "shelter_gamma", "sm_init_frac", "sm_daily_loss", "gamma_pt",
    "brutsaert_coef", "lsurr_fallback_eps", "refl_fallback_albedo", "veg_lw_transmitted", "lsurr_ray_weights",
    "soil_water_v2", "open_water_class", "et_stress_frac", "canopy_exchange_tau", "wind_height_agl" };

const char* const kMaterialNames[NUM_MATERIALS] = {
    "average ground", "asphalt", "soil", "grass", "cobble", "shrubs and trees", "concrete", "clay tile roof",
    "membrane roof", "metal roof", "wood", "wall", "artificial grass", "water", "average roof" };

ModelParams default_params() {
    /* Calibrated against 220 Landsat 8/9 LST scenes of central Maribor (2013-2025): hc_scale, the seasonal
       storage term (amplitude, phase), and factors on OHM a1 of the impervious classes and on alpha_PT of the
       pervious classes. Classes 0 (average ground) and 14 (average roof) are area-weighted means of the
       calibrated classes, for runs without a material map:
         average ground = 0.614 asphalt + 0.187 grass + 0.185 soil + 0.013 concrete
         average roof   = 0.433 clay tile + 0.334 concrete + 0.170 metal + 0.063 membrane */
    static const float p[P_NUM] = {
        1.5636f, 4.0f, 0.5f, 0.32f, 10.0f, 70.910f, 359.02f,
        0.5f, 0.3f, 0.6f, 6.0f, 0.5f, 0.05f, 0.066f,
        1.24f, 0.95f, 0.2f, 1.0f, 1.0f,
        1.0f, 13.0f, 0.5f, 1.0f, 1.0f };

    // albedo, emissivity, OHM a1, a2 (h), a3 (W m-2), Priestley-Taylor alpha, field capacity (mm)
    static const float m[NUM_MATERIALS][M_NUM] = {
        { 0.15368f, 0.94561f, 0.61629f, 0.35141f, -38.285f, 0.28111f,  55.756f }, //  0 average ground
        { 0.13f,    0.95f,    0.78406f, 0.3f,     -42.42f,  0.0f,       0.0f }, //  1 asphalt
        { 0.17f,    0.95f,    0.355f,   0.335f,   -35.275f, 0.70074f, 200.0f }, //  2 soil
        { 0.21f,    0.93f,    0.32f,    0.54f,    -27.4f,   0.80854f, 100.0f }, //  3 grass
        { 0.28f,    0.9f,     0.78406f, 0.3f,     -42.42f,  0.0f,       0.0f }, //  4 cobble
        { 0.17f,    0.98f,    0.215f,   0.325f,   -19.85f,  0.92713f, 150.0f }, //  5 shrubs and trees
        { 0.23f,    0.9f,     0.67272f, 0.3f,     -42.42f,  0.0f,       0.0f }, //  6 concrete
        { 0.27f,    0.9f,     0.31316f, 0.337f,   -33.87f,  0.0f,       0.0f }, //  7 clay tile roof
        { 0.5f,     0.85f,    0.13918f, 0.337f,   -33.87f,  0.0f,       0.0f }, //  8 membrane roof
        { 0.5f,     0.85f,    0.13918f, 0.337f,   -33.87f,  0.0f,       0.0f }, //  9 metal roof
        { 0.15f,    0.9f,     0.23197f, 0.337f,   -33.87f,  0.0f,       0.0f }, // 10 wood
        { 0.3f,     0.91f,    0.78406f, 0.3f,     -42.42f,  0.0f,       0.0f }, // 11 wall
        { 0.1f,     0.93f,    0.32f,    0.54f,    -27.4f,   0.1f,      50.0f }, // 12 artificial grass
        { 0.1f,     0.95f,    0.5f,     0.21f,    -39.1f,   1.26f,   1000.0f }, // 13 water
        { 0.31023f, 0.88835f, 0.39272f, 0.32464f, -36.726f, 0.0f,       0.0f }, // 14 average roof
    };

    ModelParams mp;
    std::memcpy(mp.p, p, sizeof(p));
    std::memcpy(mp.materials, m, sizeof(m));
    return mp;
}

bool read_params(const std::string& path, ModelParams& mp, std::string& log) {
    std::ifstream ifd(path);
    if (!ifd.is_open()) { log += "params: cannot open " + path + "\n"; return false; }
    std::string line;
    while (std::getline(ifd, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream is(line);
        std::string key;
        if (!(is >> key)) continue;
        if (key == "class") {
            int c; float v[M_NUM];
            if (is >> c >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6] && c >= 0 && c < NUM_MATERIALS)
                for (int k = 0; k < M_NUM; ++k) mp.materials[c][k] = v[k];
            else log += "params: bad class line: " + line + "\n";
            continue;
        }
        bool found = false;
        for (int i = 0; i < P_NUM; ++i)
            if (key == kParamNames[i]) { is >> mp.p[i]; found = true; break; }
        if (!found) log += "params: unknown key " + key + "\n";
    }
    return true;
}

void write_params(const std::string& path, const ModelParams& mp) {
    std::ofstream o(path);
    o << "# LSTGPU model constants used in this run (--params format)\n";
    for (int i = 0; i < P_NUM; ++i) o << kParamNames[i] << " " << mp.p[i] << "\n";
    o << "# class i albedo emissivity a1 a2 a3 alpha_pt field_capacity_mm\n";
    for (int c = 0; c < NUM_MATERIALS; ++c) {
        o << "class " << c;
        for (int k = 0; k < M_NUM; ++k) o << " " << mp.materials[c][k];
        o << "   # " << kMaterialNames[c] << "\n";
    }
}

} // namespace lstgpu
