#include "cuda/kernels/seb_kernel.cuh"

#include "cuda/physics/canopy.cuh"
#include "cuda/physics/convection.cuh"
#include "cuda/physics/latent_heat.cuh"
#include "cuda/physics/sky_longwave.cuh"
#include "cuda/physics/storage_heat.cuh"
#include "cuda/physics/wind_profile.cuh"

__global__ void seb_kernel(const unsigned int* grid, const float2* grid_s_cover_veg, float* targetData, int n, int h, int start_h) {
    const unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id >= num_data) return;

    float F_svf = targetData[SVF * num_data + t_id];

    int ind = n * num_hours + h;

    float thermal = 0.0f;

    int cc = targetData[CLASSTYPE * num_data + t_id];

    float I_r = targetData[REFLECTIVE_IRD_H * num_data + t_id] + targetData[REFLECTIVE_IRD_H2 * num_data + t_id];
    float I = targetData[SOLAR_POTENTIAL_H * num_data + t_id] + I_r; // incident shortwave
    float slope = targetData[SLOPE * num_data + t_id];

    // soil water bucket (mm), initialised to a fraction of field capacity
    float SM = targetData[SOIL_MOISTURE * num_data + t_id];
    if (SM == -1.0f) {
        SM = targetData[SOIL_MOISTURE * num_data + t_id] = P_g[P_SM_INIT] * class_matrix[cc][M_FIELD_CAP];
    }
    if (h == start_h && P_g[P_SOIL_V2] <= 0.0f) { // simple bucket: daily drainage
        SM *= (1.0f - P_g[P_SM_LOSS]);
    }

    float conductive_Idiff = 0;

    float pw = p_w[ind]; // water vapour pressure (Pa)
    float cf = c_f[ind]; // cloud cover fraction
    float E = class_matrix[cc][M_EMISSIVITY];

    float Ta = ambient_temperature[ind] + 273.15;

    float a = class_matrix[cc][M_ALBEDO];

    cf = fminf(fmaxf(cf, 0.0f), 1.0f);

    float cf2 = cloud_fraction_from_components(direct_irad_meas[ind], diffuse_irad_meas[ind], cosf((90.0f - sun_azm_alt[ind].y) * DTR));

    if (cf2 >= 0.0f) cf = cf2;

    F_svf = fminf(fmaxf(F_svf, 0.0f), 1.0f);
    Ta = fmaxf(Ta, 150.0f); // K

    // ---- longwave from the sky (W m-2) ----
    float eps_sky = eps_sky_walton_cf(brutsaert_emissivity(pw, Ta), cf); // clear sky with the cloud correction
    float view_factor = tilted_sky_view(F_svf, slope);

    // the canopy intercepts f_c of the sky seen through it and emits at air temperature, as in the exchange sums (Lsurr)
    const float f_c_sky = canopy_fraction(n);
    const float F_sky_veg = fminf(fmaxf(grid_s_cover_veg[t_id].x / total_view_cover, 0.0f), F_svf);
    float Lsky = SIGMA * Ta * Ta * Ta * Ta * (eps_sky * (view_factor - f_c_sky * F_sky_veg) + P_g[P_LSURR_FB_EPS] * f_c_sky * F_sky_veg);

    unsigned int g = targetData[(NUM_ATTRIBUTES - 1) * num_data + t_id];
    unsigned int gh = grid[(int)(g % width) + (int)(((float)g / (float)width)) * width];
    float z_veg_top = GRID_ZV(gh);

    float Lsurr = targetData[LONGWAVE_SURR_H * num_data + t_id];
    float cell_z = targetData[CELL_HEIGHT * num_data + t_id];

    if (Lsurr == 0.0f) { // no sampled surroundings: surroundings at air temperature
        Lsurr = P_g[P_LSURR_FB_EPS] * SIGMA * Ta * Ta * Ta * Ta * (1.0f - view_factor);
    }

    if (I_r == 0.0f) { // no sampled surroundings: mean urban albedo
        I_r = P_g[P_REFL_FB_ALB] * I * (1.0f - view_factor);
        I += I_r; // the fallback also enters the energy balance
    }

    // ---- effective local wind speed ----
    float W_f = targetData[WIND_ZD * num_data + t_id]; // shelter factor
    // profile height: above the local ground (wind_height_agl 1) or above the height reference (0)
    const float z_wind = (P_g[P_WIND_AGL] > 0.5f) ? targetData[HEIGHT_AGL * num_data + t_id] : cell_z;
    float Vf = W_f * profile_wind_speed(wind_speed[ind], z_wind);
    Vf = fmaxf(Vf, P_g[P_WIND_MIN]);

    if (z_veg_top > 0.01f && (cell_z + 0.5f) < z_veg_top) { // inside a canopy
        Vf *= canopy_wind_factor(n);
    }

    // ---- storage heat flux, objective hysteresis model (OHM) ----
    float Q_net_old = targetData[NET_RAD_OLD * num_data + t_id];

    float T_src = targetData[SURFACE_TEMPERATURE_H * num_data + t_id]; // LST of the previous hour

    float L_up_approx;
    if (T_src == 9999) L_up_approx = E * SIGMA * Ta * Ta * Ta * Ta; // first hour: approximate with Ta
    else {
        T_src += 273.15;
        L_up_approx = E * SIGMA * T_src * T_src * T_src * T_src;
    }

    float Q_net = (1.0f - a) * I + E * (Lsky + Lsurr) - L_up_approx;

    if (Q_net_old == 9999) Q_net_old = Q_net;
    conductive_Idiff = (Q_net - Q_net_old);

    float seasonal_a3 = seasonal_storage(n);

    float Qs = ohm_storage(cc, Q_net, conductive_Idiff) + seasonal_a3;

    // ---- latent heat flux, Priestley & Taylor (1972) ----
    float alpha_pt = class_matrix[cc][M_ALPHA_PT];
    float QE = 0.0f;

    float FC = class_matrix[cc][M_FIELD_CAP]; // field capacity (mm)
    float rain_mm = precip_meas[ind];         // mm per time step

    float gamma_pt = P_g[P_GAMMA_PT];

    // PT-JPL green-canopy constraint (Fisher et al. 2008): transpiration scales with the canopy fraction
    // f_c = 1 - exp(-k LAI(n)), the same Beer-Lambert form as the canopy transmission
    float alpha_eff = alpha_pt;
    if (cc == 5) { // shrubs and trees: deciduous phenology
        alpha_eff = alpha_pt * canopy_fraction(n);
    }

    // soil_water_v2: the bucket and a gradual water stress limit the latent heat also in the iteration below,
    // the bucket is depleted by the final latent heat, and open water never loses content
    const bool sw_v2 = P_g[P_SOIL_V2] > 0.0f;
    const bool open_water = sw_v2 && P_g[P_OPEN_WATER] >= 0.0f && cc == (int)P_g[P_OPEN_WATER];
    float et_beta = 1.0f, qe_max = 1e30f, mm_to_W = 1.0f;

    if (FC > 0.0f && alpha_eff > 0.0f) { // materials with water storage
        SM = fminf(SM + rain_mm, FC); // rain beyond field capacity runs off
        if (open_water) SM = FC;

        // potential evapotranspiration
        float Delta = saturation_slope(Ta - 273.15f);

        float avail_energy;

        if (T_src == 9999) {
            avail_energy = (1.0f - a) * I + E * (Lsky + Lsurr - SIGMA * Ta * Ta * Ta * Ta) - Qs;
        }
        else {
            avail_energy = (1.0f - a) * I + E * (Lsky + Lsurr - SIGMA * T_src * T_src * T_src * T_src) - Qs;
        }

        float PET = priestley_taylor(alpha_eff, Delta, gamma_pt, avail_energy); // W m-2
        PET = fmaxf(0.0f, PET);

        const float lambda = latent_heat_vaporization(Ta);
        const float rho_w = 1000.0f;
        float time_step_s = time_inter * 3600.0f;
        float PET_mm = (PET * time_step_s) / (lambda * rho_w) * 1000.0f; // mm per time step

        // actual ET limited by the available water
        float ETa_mm = fminf(PET_mm, SM);
        if (sw_v2) {
            mm_to_W = lambda * rho_w / (1000.0f * time_step_s); // W m-2 per mm of water per time step
            if (!open_water && P_g[P_ET_STRESS] > 0.0f) et_beta = fminf(1.0f, SM / (P_g[P_ET_STRESS] * FC));
            qe_max = open_water ? 1e30f : SM * mm_to_W; // the bucket can't give more than it holds
            ETa_mm = open_water ? PET_mm : fminf(PET_mm * et_beta, SM);
        }

        QE = (ETa_mm / 1000.0f) * (lambda * rho_w) / time_step_s;

        if (!sw_v2) SM -= ETa_mm; // soil_water_v2: depleted by the latent heat of the energy balance, below

        SM = fmaxf(0.0f, SM);
        targetData[SOIL_MOISTURE * num_data + t_id] = SM;
    }
    else {
        QE = 0.0f;
    }

    // ---- sensible heat: forced and natural convection ----
    float cos_phi = fabsf(cosf(slope)); // 1 = horizontal, 0 = vertical

    float h_c_for = forced_convection(Vf, cos_phi);
    Vf = fmaxf(Vf, P_g[P_WIND_MIN]); // the reported wind speed

    float h_c_nat = 0.0f, h_c;

    float lst0;

    // start from the previous hour's LST; in the first hour from a linearised balance with Ts - Ta = 10 K
    if (T_src == 9999) {
        h_c_nat = natural_convection_unstable(10.0f, cos_phi);
        h_c = heat_transfer_coefficient(h_c_for, h_c_nat);

        QE = fminf(QE, fmaxf((1.0f - a) * I + E * (Lsky + Lsurr) - (E * SIGMA * Ta * Ta * Ta * Ta) - Qs, 0.0f));

        lst0 = Ta + ((1.0f - a) * I + E * (Lsky + Lsurr) - Qs - QE - E * SIGMA * Ta * Ta * Ta * Ta) / (h_c + 4.0f * E * SIGMA * Ta * Ta * Ta);
        lst0 = fminf(fmaxf(lst0, 220.0f), 350.0f);
    }
    else {
        lst0 = T_src;
    }

    if (lst0 != Ta) h_c_nat = natural_convection(lst0, Ta, cos_phi); // at Ts = Ta the start value's h_c_nat stays
    h_c = heat_transfer_coefficient(h_c_for, h_c_nat);

    float h_c_flux = h_c;

    float c = -E * (Lsky + Lsurr) - h_c_flux * Ta - (1 - a) * I + Qs + QE;
    float y = lst0 * lst0 * lst0;
    float lst = (3 * E * SIGMA * y * lst0 - c) / (4 * E * SIGMA * y + h_c); // first Newton step

    // iteration with Ta held fixed
    for (int i = 0; i < 50; ++i) {
        lst0 = lst;

        h_c_nat = natural_convection(lst0, Ta, cos_phi);
        h_c = heat_transfer_coefficient(h_c_for, h_c_nat);

        h_c_flux = h_c;

        // net radiation and storage at the current LST
        Q_net = (1.0f - a) * I + E * (Lsky + Lsurr) - (E * SIGMA * lst * lst * lst * lst);
        Qs = ohm_storage(cc, Q_net, Q_net - Q_net_old) + class_matrix[cc][M_OHM_A3] + seasonal_a3;

        if (alpha_eff > 0.0f) {
            float Delta = saturation_slope(Ta - 273.15f);

            float avail_energy = Q_net - Qs;
            QE = priestley_taylor(alpha_eff, Delta, gamma_pt, avail_energy);
            QE = fmaxf(0.0f, fminf(QE, avail_energy));

            if (sw_v2) // water stress and the bucket; no cap by the sensible heat
                QE = fminf(QE * et_beta, qe_max);
            else
                QE = fminf(QE, fmaxf(((1.0f - a) * I                    // net shortwave
                    + E * (Lsky + Lsurr)                                 // incoming longwave
                    - (E * SIGMA * lst * lst * lst * lst)                // emitted longwave
                    - Qs                                                 // storage
                    - h_c_flux * (lst - Ta)                              // sensible heat
                    ), 0.0f));
        }
        else {
            QE = 0.0f;
        }

        c = -E * (Lsky + Lsurr) - h_c_flux * Ta - (1.0f - a) * I + Qs + QE;
        y = lst0 * lst0 * lst0;

        float lst_new = (3.0f * E * SIGMA * y * lst0 - c) / (4.0f * E * SIGMA * y + h_c);
        lst = 0.5f * lst0 + 0.5f * lst_new; // 50 % under-relaxation

        if (fabsf(lst - lst0) < 0.001f) {
            break;
        }
    }

    thermal = (lst - 273.15);
    Ta = (Ta - 273.15);

    targetData[NET_RAD_OLD * num_data + t_id] = Q_net;
    targetData[SURFACE_TEMPERATURE_H * num_data + t_id] = thermal;

    if (sw_v2 && FC > 0.0f && alpha_eff > 0.0f) { // deplete the bucket by the latent heat of the energy balance
        SM = open_water ? FC : fmaxf(0.0f, SM - QE / mm_to_W);
        targetData[SOIL_MOISTURE * num_data + t_id] = SM;
    }

    targetData[REFLECTIVE_IRD_SEB_H * num_data + t_id] = I_r;
    targetData[LONGWAVE_SKY_H * num_data + t_id] = Lsky;
    targetData[LONGWAVE_SURR_SEB_H * num_data + t_id] = Lsurr;
    targetData[NET_RAD_H * num_data + t_id] = Q_net;
    targetData[STORAGE_H * num_data + t_id] = Qs;
    targetData[LATENT_H * num_data + t_id] = QE;
    targetData[SENSIBLE_H * num_data + t_id] = h_c_flux * (thermal - Ta); // both in degC here
    targetData[H_C_H * num_data + t_id] = h_c;
    targetData[WINDSPEED_H * num_data + t_id] = Vf;
}
