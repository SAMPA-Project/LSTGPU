#pragma once

/* Geometry of the hemisphere direction bins (azimuth x elevation, unit sphere) on the host: the solid angle of a bin,
   the cosine-weighted hemisphere that normalises the sky view factor, and the circumsolar region of an hour.
   Angles in radians unless the name says degrees. No CUDA dependency. */

namespace lstgpu {

// Area of a spherical triangle from its spherical excess; points as (latitude, longitude)
float spherical_excess(float P1_rad[2], float P2_rad[2], float P3_rad[2]);

// Solid angle of the bin between latitudes lat1 < lat2 and longitudes lon1 < lon2
float bin_solid_angle(float lat1, float lon1, float lat2, float lon2);

/* Cosine-weighted solid angle of the upper hemisphere (sin of the elevation of the bin centre, as the rays: the
   visible sky of an unobstructed horizontal voxel sums to the same value, so that its sky view factor is 1), and the
   solid angle of its lowest elevation band. hres, vres: bin size (degrees); hsize, vsize: bin counts. */
void hemisphere_cover(unsigned int hsize, unsigned int vsize, float hres, float vres, float& total_view, float& total_horizon);

/* Circumsolar region of one hour: bins[j + i * vsize] = the solid angle of the bin of the sun at (sun_azimuth,
   sun_altitude) degrees and of every upper-hemisphere bin within csa degrees of it, 0 elsewhere. */
void circumsolar_region(float sun_azimuth, float sun_altitude, unsigned int hsize, unsigned int vsize, float hres,
                        float vres, float csa, float* bins);

} // namespace lstgpu
