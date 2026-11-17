#pragma once

/* Diffuse irradiance on an inclined surface by the modified anisotropic Perez model (Perez et al. 1990): the
   isotropic, circumsolar and horizon parts of the sky, each weighted by the part of it the surface sees. */

#include "cuda/device_globals.cuh"

/* cos_z_angle: cosine of the solar zenith angle; dif_irad, dir_irad: diffuse and direct horizontal irradiance
   (W m-2); n: day; cos_inc: cosine of the angle of incidence of the sun on the surface (also when the sun is
   obstructed); hviewshed_cover: cosine-weighted visible sky (isotropic part); horizont_cover: cosine-weighted visible
   share of the front half of the horizon band; circumsolar_cover: visible share of the circumsolar region. */
__device__ inline float perez_diffuse(const float& cos_z_angle, const float& dif_irad, const float& dir_irad, const float& slope, const int& n,
    const float& cos_inc, const float hviewshed_cover, const float horizont_cover, const float& circumsolar_cover) {
    // relative optical air mass
    float z_deg = acosf(cos_z_angle) * (180.0f / PI);
    float AM = 1.0f / (cos_z_angle + 0.50572f * powf(96.07995f - z_deg, -1.6364f)); // Kasten-Young (1989)

    // zenith angle
    float z_angle = acosf(cos_z_angle);

    // Perez sky clearness
    float eps = ((dif_irad + dir_irad / cosf(z_angle)) / dif_irad + 1.041f * (z_angle * z_angle * z_angle)) / (1.0f + 1.041f * z_angle * z_angle * z_angle);

    // Perez sky brightness
    float delta = (AM * dif_irad) / E0[n];

    unsigned int eps_ind = 7;
    if (eps >= 1.0f && eps < 1.065f) eps_ind = 0;
    else if (eps < 1.230f) eps_ind = 1;
    else if (eps < 1.500f) eps_ind = 2;
    else if (eps < 1.950f) eps_ind = 3;
    else if (eps < 2.800f) eps_ind = 4;
    else if (eps < 4.500f) eps_ind = 5;
    else if (eps < 6.200f) eps_ind = 6;

    // F1 = circumsolar brightness, F2 = horizon brightness
    float F1 = fmaxf(0.0f, F_matrix[0][eps_ind] + F_matrix[1][eps_ind] * delta + F_matrix[2][eps_ind] * z_angle);
    float F2 = F_matrix[3][eps_ind] + F_matrix[4][eps_ind] * delta + F_matrix[5][eps_ind] * z_angle;

    // the isotropic, circumsolar and horizon parts are weighted by their visible fractions; the circumsolar part scales
    // with a/b = max(0, cos theta) / max(cos 85, cos z) and the horizon brightening with sin(slope), as in Perez et al. (1990)
    const float ab = fmaxf(cos_inc, 0.0f) / fmaxf(cos_z_angle, 0.08715574274f);
    float dif = dif_irad * ((1 - F1) * hviewshed_cover + F1 * circumsolar_cover * ab + F2 * horizont_cover * sinf(slope));

    return(dif);
}
