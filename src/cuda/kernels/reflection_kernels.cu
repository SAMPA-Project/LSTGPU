#include "cuda/kernels/reflection_kernels.cuh"

#include "cuda/physics/canopy.cuh"
#include "cuda/physics/surface_geometry.cuh"

__global__ void surroundings_kernel(float* targetData, int n, int h, const int* svf_ind, bool reflect_order) {
    const unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id >= num_data) return;

    float slope = targetData[SLOPE * num_data + t_id];
    float3 Cdir;

    unsigned int g = targetData[(NUM_ATTRIBUTES - 1) * num_data + t_id];
    float init_z = fmaxf(targetData[CELL_HEIGHT * num_data + t_id], 0.1f), src_z;
    int init_x, init_y;
    init_x = g % width;
    init_y = ((float)g / (float)width);

    float len;

    float aspect = targetData[ASPECT * num_data + t_id];

    const float3 receiver_normal = surface_normal(slope, aspect);

    float T_s = 0, I_r = 0; // emitted longwave, reflected shortwave
    int src_id = 0;
    int src_x, src_y;

    int classsrc;
    float emission_src;
    float albedo_src;
    float I_src;
    float T_src;

    float cos_theta_r;
    float dist2;

    float F_sum = 0.0; // sum of the view factors
    const float area = gridRes * gridRes;

    float VegTerm = 1.0f;
    for (unsigned int i = 0; i < max_indices_g; i++) {
        src_id = svf_ind[i * num_data + t_id];

        if (src_id == 0) break;

        if (src_id < 0) { // seen through a canopy
            src_id = -src_id;
            // canopy_exchange_tau 1 weights the source with the transmittance tau = exp(-k LAI), as the direct beam;
            // 0 with the canopy fraction 1 - tau
            VegTerm = (P_g[P_CANOPY_TAU] > 0.0f) ? canopy_transmittance(n) : canopy_fraction(n);
        }
        else {
            VegTerm = 1.0f;
        }
        const float hits = (float)((src_id >> SVF_CNT_SHIFT) + 1); // rays that hit this voxel (1 without lsurr_ray_weights)
        src_id &= SVF_ID_MASK;

        src_z = fmaxf(targetData[CELL_HEIGHT * num_data + src_id], 0.1f);
        slope = targetData[SLOPE * num_data + src_id];
        aspect = targetData[ASPECT * num_data + src_id];
        classsrc = (int)targetData[CLASSTYPE * num_data + src_id];
        src_x = (int)targetData[(NUM_ATTRIBUTES - 1) * num_data + src_id] % width;
        src_y = ((float)targetData[(NUM_ATTRIBUTES - 1) * num_data + src_id] / (float)width);

        // direction from the receiver to the source
        Cdir.x = (src_x - init_x) * gridRes;
        Cdir.y = (src_y - init_y) * gridRes;
        Cdir.z = (src_z - init_z) * gridRes;
        len = sqrtf(Cdir.x * Cdir.x + Cdir.y * Cdir.y + Cdir.z * Cdir.z);

        dist2 = fmaxf(len * len, gridRes * gridRes);

        Cdir.x /= len;
        Cdir.y /= len;
        Cdir.z /= len;

        float cos_theta_i = fmaxf(0.0f, receiver_normal.x * (Cdir.x) +
            receiver_normal.y * (Cdir.y) +
            receiver_normal.z * (Cdir.z));

        if (len == 0.0f) continue;
        if (cos_theta_i <= 0.0f) continue;

        const float3 source_normal = surface_normal(slope, aspect);
        cos_theta_r = source_normal.x * (-Cdir.x) +
            source_normal.y * (-Cdir.y) +
            source_normal.z * (-Cdir.z);

        if (cos_theta_r <= 0.0f) continue; // the source faces away from the receiver

        albedo_src = class_matrix[classsrc][M_ALBEDO];
        if (reflect_order == 0)
            I_src = targetData[SOLAR_POTENTIAL_H * num_data + src_id];
        else
            I_src = targetData[REFLECTIVE_IRD_H * num_data + src_id];

        float F_pair;
        if (P_g[P_LSURR_RAYW] > 0.0f) {
            // each ray stands for its direction bin: rays x bin solid angle x cos / pi
            const float e = asinf(fminf(fmaxf(Cdir.z, -1.0f), 1.0f)) / DTR + 90.0f;
            const float lat1 = fminf(floorf(e / vres_f), (float)vsize_f - 1.0f) * vres_f - 90.0f;
            const float omega = hres * DTR * (sinf((lat1 + vres_f) * DTR) - sinf(lat1 * DTR));
            F_pair = hits * omega * cos_theta_i / PI;
        }
        else
            F_pair = (cos_theta_i * cos_theta_r) * (area / (PI * dist2)); // 1 m2 form factor
        F_sum += (float)F_pair;

        I_r += albedo_src * I_src * F_pair * VegTerm;

        if (reflect_order == 0) {
            T_src = targetData[SURFACE_TEMPERATURE_H * num_data + src_id];
            emission_src = class_matrix[classsrc][M_EMISSIVITY];

            float Ta = ambient_temperature[n * num_hours + h] + 273.15f;
            // source not simulated yet (first hour): Ta + radiative excess
            T_src = (T_src == 9999.0f) ? Ta + ((1.0f - albedo_src) * (diffuse_irad_meas[n * num_hours + h] + direct_irad_meas[n * num_hours + h])) / (4.0f * emission_src * SIGMA * powf(Ta + 10.0f, 3.0f)) : (T_src + 273.15f);

            T_s += emission_src * SIGMA * (T_src * T_src * T_src * T_src) * F_pair * VegTerm;
            if (P_g[P_VEG_LW_TRANS] > 0.0f && VegTerm < 1.0f) // radiation passing through the canopy: surfaces behind at air temperature
                T_s += P_g[P_LSURR_FB_EPS] * SIGMA * (Ta * Ta * Ta * Ta) * F_pair * (1.0f - VegTerm);
        }
    }

    // the cached view factors cover the obstructed part of the view, 1 - SVF (both cosine-weighted: the scale only closes
    // the sky/surroundings partition for rays that miss or face away)
    float scale = 1.0f;
    if (F_sum > 0.0) {
        scale = fmaxf(0.0f, 1.0f - targetData[SVF * num_data + t_id]) / F_sum;
    }

    T_s *= scale;
    I_r *= scale;

    if (reflect_order == 0) {
        if (T_s > 0) {
            targetData[LONGWAVE_SURR_H * num_data + t_id] += T_s;
        }
        if (I_r > 0) {
            targetData[REFLECTIVE_IRD_H * num_data + t_id] += I_r;
        }
    }
    else {
        if (I_r > 0) {
            targetData[REFLECTIVE_IRD_H2 * num_data + t_id] += I_r;
        }
    }
}
