#include "cuda/kernels/irradiance_kernel.cuh"

#include "cuda/physics/canopy.cuh"
#include "cuda/physics/perez_diffuse.cuh"
#include "cuda/physics/surface_geometry.cuh"

__global__ void shortwave_kernel(const grid_s_type* grid_s, const grid_s_type* grid_s_veg, const float2* grid_s_cover,
    const float2* grid_s_cover_veg, const float* grid_s_front_hz, const float* sun_circumsolar_region, float* targetData,
    int n, int h) {
    const unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id >= num_data) return;

    float slope = targetData[SLOPE * num_data + t_id];
    // sky view factor: cosine-weighted visible sky over the cosine-weighted hemisphere of a horizontal plane
    // (total_view_cover); an unobstructed inclined plane gets (1 + cos(slope)) / 2, which also bounds the discretised sum
    float F_sky = fminf(fmaxf(grid_s_cover[t_id].x / total_view_cover, 0.0f), (1.0f + cosf(slope)) / 2.0f);
    targetData[SVF * num_data + t_id] = F_sky;

    float z_angle, angle_of_incidence;
    int ind = n * num_hours + h;
    float Ib_c_i = 0, Id_c_i = 0; // direct, diffuse

    float S_i = 0;

    float patch_area, total_area, circumsolar_cover, circumsolar_cover_veg;
    float f_c = canopy_fraction(n); // canopy interception
    float aspect = targetData[ASPECT * num_data + t_id];

    if (sun_azm_alt[ind].y >= 1) {
        // sun behind terrain or buildings?
        grid_s_type base = (grid_s_type)floor(sun_azm_alt[ind].x / (float)hres);
        grid_s_type val = horizon_get(t_id, base, grid_s);
        if ((grid_s_type)floor(sun_azm_alt[ind].y / (float)vres) <= val)
            S_i = 1;

        // visible part of the circumsolar region
        circumsolar_cover = 0.0f;
        circumsolar_cover_veg = 0.0f;
        total_area = 0.0f;

        for (int i = 0; i < hsize; ++i) {
            grid_s_type val_ter = horizon_get(t_id, (grid_s_type)i, grid_s);
            grid_s_type val_veg = horizon_get(t_id, (grid_s_type)i, grid_s_veg);

            for (int j = 0; j < vsize; ++j) {
                patch_area = sun_circumsolar_region[j + i * vsize + h * vsize * hsize];
                if (patch_area > 0.0f) {
                    total_area += patch_area;
                    if ((grid_s_type)j <= val_ter && val_ter > 0)
                        circumsolar_cover += patch_area;     // blocked by terrain/building
                    else if ((grid_s_type)j <= val_veg && val_veg > 0)
                        circumsolar_cover_veg += patch_area; // blocked by vegetation only
                }
            }
        }

        if (total_area > 0.0f)
            circumsolar_cover = 1.0f - circumsolar_cover / total_area
            - f_c * (circumsolar_cover_veg / total_area);
        else
            circumsolar_cover = (S_i != 1) ? 1.0f : 0.0f; // no patch data: fall back to sun visibility

        circumsolar_cover = fminf(fmaxf(circumsolar_cover, 0.0f), 1.0f);
        z_angle = cosf((90.0f - sun_azm_alt[ind].y) * DTR); // cos of the zenith angle

        if (S_i != 1 && (acosf(z_angle) > 0.08715574274f && acosf(z_angle) < 1.55334)) {
            angle_of_incidence = cos_incidence(z_angle, slope, aspect, sun_azm_alt[ind].x);

            if (angle_of_incidence > 0.0f) {
                val = horizon_get(t_id, base, grid_s_veg);
                if ((grid_s_type)floor(sun_azm_alt[ind].y / (float)vres) <= val) { // sun behind a canopy
                    S_i = 1;
                    Ib_c_i = fmaxf(direct_irad_meas[ind], 1e-3f) * (angle_of_incidence / z_angle) * (1.0f - f_c);
                }
                else {
                    Ib_c_i = fmaxf(direct_irad_meas[ind], 1e-3f) * (angle_of_incidence / z_angle);
                }
            }
        }

        // angle of incidence for the circumsolar part of the diffuse irradiance, also when the direct beam is obstructed
        const float cos_inc = cos_incidence(z_angle, slope, aspect, sun_azm_alt[ind].x);
        const float front_hz = grid_s_front_hz[t_id];
        const float V_hz = (front_hz > 0.0f) ? fminf(fmaxf((grid_s_cover[t_id].y - f_c * grid_s_cover_veg[t_id].y) / front_hz, 0.0f), 1.0f) : 0.0f;

        Id_c_i = perez_diffuse(z_angle, fmaxf(diffuse_irad_meas[ind], 1e-3f), fmaxf(direct_irad_meas[ind], 1e-3f), slope, n, cos_inc,
            F_sky - f_c * (grid_s_cover_veg[t_id].x / total_view_cover),
            V_hz,
            circumsolar_cover
        );
        if (Id_c_i < 0.0f) Id_c_i = 0.0f;
    }

    targetData[SHADOWING_H * num_data + t_id] = S_i;
    targetData[DIRECT_IRD_H * num_data + t_id] = Ib_c_i;
    targetData[DIFFUSE_IRD_H * num_data + t_id] = Id_c_i;
    targetData[SOLAR_POTENTIAL_H * num_data + t_id] = (Ib_c_i + Id_c_i);
    targetData[LONGWAVE_SURR_H * num_data + t_id] = 0;
    targetData[REFLECTIVE_IRD_H * num_data + t_id] = 0;
    targetData[REFLECTIVE_IRD_H2 * num_data + t_id] = 0;
}
