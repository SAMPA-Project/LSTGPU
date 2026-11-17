#include "cuda/device_upload.cuh"

#include <cmath>
#include <vector>

#include "cuda_utils.cuh"
#include "cuda/device_globals.cuh"

namespace lstgpu {

void upload_model(const ModelParams& mp) {
    cudaSafeCall(cudaMemcpyToSymbol(P_g, mp.p, sizeof(float) * P_NUM, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(class_matrix, mp.materials, sizeof(float) * NUM_MATERIALS * M_NUM, 0, cudaMemcpyHostToDevice));

    float hF_matrix[6][8] = { // Perez F matrix
        { -0.0083117f ,	0.1299457f , 0.3296958f , 0.5682053f , 0.8730280f , 1.1326077f , 1.0601591f , 0.6777470f }, // f11
        { 0.5877285f ,	0.6825954f,	0.4868735f , 0.1874525f , -0.3920403f ,	-1.2367284f , -1.5999137f ,	-0.3272588f }, // f12
        { -0.0620636f ,	-0.1513752f , -0.2210958f ,	-0.2951290f , -0.3616149f, - 0.4118494f , -0.3589221f , -0.2504286f }, // f13
        { -0.0596012f ,	-0.0189325f , 0.0554140f ,	0.1088631f , 0.2255647f , 0.2877813f ,	0.2642124f , 0.1561313f }, // f21
        { 0.0721249f 	,	0.0659650f , -0.0639588f ,	-0.1519229f , -0.4620442f ,	-0.8230357f , -1.1272340f ,	-1.3765031f }, // f22
        { -0.0220216f ,	-0.0288748f , -0.0260542f ,	-0.0139754f , 0.0012448f , 0.0558651f , 0.1310694f , 0.2506212f } // f23
    };
    cudaSafeCall(cudaMemcpyToSymbol(F_matrix, hF_matrix, sizeof(float) * 6 * 8, 0, cudaMemcpyHostToDevice));
}

void upload_forcing(const Forcing& forcing, const std::vector<SunSample>& sun, int num_hours) {
    // extraterrestrial irradiance per day
    float E0_tmp[365];
    for (int n = 0; n < 365; ++n) {
        float B = 2 * PI * (n / 365.0f);
        E0_tmp[n] = 1361 * (1.00011f + 0.034221f * cos(B) + 0.00128f * sin(B) + 0.000719f * cos(2 * B) + 0.000077f * sin(2 * B));
    }
    cudaSafeCall(cudaMemcpyToSymbol(E0, E0_tmp, 365 * sizeof(float), 0, cudaMemcpyHostToDevice));

    std::vector<float2> tmp_sun_azm_alt(FORCING_ARRAY_SIZE, make_float2(0.0f, 0.0f));
    std::vector<float> tmp_ib(FORCING_ARRAY_SIZE, 0.0f), tmp_id(FORCING_ARRAY_SIZE, 0.0f), tmp_ta(FORCING_ARRAY_SIZE, 0.0f);
    std::vector<float> tmp_pw(FORCING_ARRAY_SIZE, 0.0f), tmp_ws(FORCING_ARRAY_SIZE, 0.0f), tmp_wd(FORCING_ARRAY_SIZE, 0.0f);
    std::vector<float> tmp_cf(FORCING_ARRAY_SIZE, 0.0f), tmp_precip_meas(FORCING_ARRAY_SIZE, 0.0f), tmp_lai(365, 0.0f);

    for (int i = 0; i < 365 * num_hours; ++i) {
        tmp_ib[i] = fmax(forcing.direct[i], 0.0f);  // direct horizontal irradiance (W m-2)
        tmp_id[i] = fmax(forcing.diffuse[i], 0.0f); // diffuse horizontal irradiance (W m-2)

        tmp_sun_azm_alt[i].x = std::get<0>(sun[i]); // azimuth (degrees)
        tmp_sun_azm_alt[i].y = std::get<1>(sun[i]); // altitude (degrees)

        tmp_ta[i] = forcing.temperature[i];

        // vapour pressure (Tetens), T in degC, stored in Pa
        float RH = forcing.humidity[i];
        RH = fminf(fmaxf(RH, 0.0f), 100.0f);
        float T = tmp_ta[i];
        float e_s_hPa = 6.112f * std::exp((17.62f * T) / (243.12f + T));
        float e_a_hPa = e_s_hPa * (RH / 100.0f);
        tmp_pw[i] = fmaxf(e_a_hPa, 0.0f) * 100.0f;

        // wind: non-negative speed, direction wrapped to [0, 360)
        float ws = forcing.wind_speed[i];
        float wd = forcing.wind_dir[i];
        tmp_ws[i] = fmaxf(ws, 0.0f);
        tmp_wd[i] = fmodf(fmodf(wd, 360.0f) + 360.0f, 360.0f);

        float cf = forcing.cloud[i];
        tmp_cf[i] = fminf(fmaxf(cf, 0.0f), 1.0f);

        tmp_precip_meas[i] = forcing.rain[i]; // mm
    }
    for (int i = 0; i < 365; ++i) {
        tmp_lai[i] = fmaxf(forcing.lai[i], 0.0f);
    }

    cudaSafeCall(cudaMemcpyToSymbol(sun_azm_alt, tmp_sun_azm_alt.data(), sizeof(float2) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(direct_irad_meas, tmp_ib.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(diffuse_irad_meas, tmp_id.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(ambient_temperature, tmp_ta.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(p_w, tmp_pw.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(wind_speed, tmp_ws.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(wind_dir, tmp_wd.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(precip_meas, tmp_precip_meas.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(c_f, tmp_cf.data(), sizeof(float) * FORCING_ARRAY_SIZE, 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(light_ext_coeffs, tmp_lai.data(), sizeof(float) * 365, 0, cudaMemcpyHostToDevice));
}

} // namespace lstgpu
