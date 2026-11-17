#include "sky_bins.h"

#include <cmath>

#include "lst_constants.h"

namespace lstgpu {

float spherical_excess(float P1_rad[2], float P2_rad[2], float P3_rad[2]) {
    float d12 = 2 * asin(sqrt(sin((P2_rad[0] - P1_rad[0]) / 2) * sin((P2_rad[0] - P1_rad[0]) / 2) + cos(P1_rad[0]) * cos(P2_rad[0]) * sin((P2_rad[1] - P1_rad[1]) / 2) * sin((P2_rad[1] - P1_rad[1]) / 2)));
    float d23 = 2 * asin(sqrt(sin((P3_rad[0] - P2_rad[0]) / 2) * sin((P3_rad[0] - P2_rad[0]) / 2) + cos(P2_rad[0]) * cos(P3_rad[0]) * sin((P3_rad[1] - P2_rad[1]) / 2) * sin((P3_rad[1] - P2_rad[1]) / 2)));
    float d31 = 2 * asin(sqrt(sin((P1_rad[0] - P3_rad[0]) / 2) * sin((P1_rad[0] - P3_rad[0]) / 2) + cos(P3_rad[0]) * cos(P1_rad[0]) * sin((P1_rad[1] - P3_rad[1]) / 2) * sin((P1_rad[1] - P3_rad[1]) / 2)));

    float s = (d12 + d23 + d31) / 2;

    return (4 * atan(sqrt(tan(s / 2) * tan((s - d12) / 2) * tan((s - d23) / 2) * tan((s - d31) / 2))));
}

float bin_solid_angle(float lat1, float lon1, float lat2, float lon2) {
    float P1[2] = { lat1, lon1 };
    float P2[2] = { lat2, lon1 };
    float P3[2] = { lat2, lon2 };
    float P4[2] = { lat1, lon2 };
    return spherical_excess(P1, P2, P4) + spherical_excess(P2, P3, P4);
}

void hemisphere_cover(unsigned int hsize, unsigned int vsize, float hres, float vres, float& total_view, float& total_horizon) {
    total_view = 0.0f;
    total_horizon = 0.0f;
    for (int i = 0; i < (int)hsize; ++i) {
        for (int j = 0; j < (int)vsize; ++j) {
            float patch_area = bin_solid_angle((j)*vres * DTR, (i)*hres * DTR, (j + 1) * vres * DTR, (i + 1) * hres * DTR);

            total_view += patch_area * sinf((j + 0.5f) * vres * DTR);

            if (j == 0)
                total_horizon += patch_area;
        }
    }
}

void circumsolar_region(float sun_azimuth, float sun_altitude, unsigned int hsize, unsigned int vsize, float hres,
                        float vres, float csa, float* bins) {
    float sun_x = (int)floor(sun_azimuth / (float)hres);
    float sun_z = (int)floor(sun_altitude / (float)vres);
    for (int i = 0; i < (int)hsize; ++i) {
        for (int j = 0; j < (int)vsize; ++j) {
            float tmp = acos(cos(((j + 0.5) * vres) * DTR) * sin(((i + 0.5) * hres) * DTR) * cos((sun_z + 0.5) * vres * DTR) * sin((sun_x + 0.5) * hres * DTR) +
                -cos(((j + 0.5) * vres) * DTR) * cos(((i + 0.5) * hres) * DTR) * -cos((sun_z + 0.5) * vres * DTR) * cos((sun_x + 0.5) * hres * DTR) +
                sin(((j + 0.5) * vres) * DTR) * sin((sun_z + 0.5) * vres * DTR));

            if ((i == sun_x && j == sun_z) || tmp <= csa * DTR) // the sun's bin always (acos of its own centre can be NaN)
                bins[j + i * vsize] = bin_solid_angle((j)*vres * DTR, (i)*hres * DTR, (j + 1) * vres * DTR, (i + 1) * hres * DTR);
            else
                bins[j + i * vsize] = 0.0f;
        }
    }
}

} // namespace lstgpu
