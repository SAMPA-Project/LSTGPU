#pragma once

/* Constants shared by the host (.cpp) and device (.cu) translation units: the LiDAR surface classes of the
   height raster, the per-voxel attribute layout of the GPU state array, and the numerical constants of the
   model. The float constants keep the values the model was calibrated with, so the results don't depend on
   a platform's M_PI. */

#define DTR   0.0174532925f   // degrees to radians
#define PI    3.14159265359f
#define SIGMA 5.670374419e-8f // Stefan-Boltzmann constant (W m-2 K-4)

// ---- LiDAR surface classes of the height raster (from the LAS classes 2 ground, 5 high vegetation, 6 building) ----
#define OBJECT_C     0 // top surface: building, else vegetation, else ground
#define TERRAIN_C    1
#define VEGETATION_C 2
#define BUILDING_C   3
#define UNCLASSIFIED 4 // vegetation points removed over buildings
#define NUM_CLASSES  5

#define NUM_MATERIALS 15 // rows of the material table (model_params.h)

// ---- per-voxel attributes of the GPU state array, stored attribute-major: data[attribute * num_voxels + voxel] ----
#define SURFACE_TEMPERATURE_H 0  // LST of the last simulated hour (degC), 9999 before the first hour
#define SLOPE                 1  // radians from horizontal
#define ASPECT                2  // radians, clockwise from north
#define CLASSTYPE             3  // material class (row of the material table)
#define CELL_HEIGHT           4  // height above the height reference (m): the lowest ground point, on the 1/SCALE_VAL m lattice
#define SVF                   5  // sky view factor
#define WIND_ZD               6  // wind shelter factor of the current hour (0..1)
#define SHADOWING_H           7  // 1 = sun not visible (or behind a canopy), 0 = sunlit
#define DIRECT_IRD_H          8  // direct shortwave (W m-2)
#define DIFFUSE_IRD_H         9  // diffuse shortwave (W m-2)
#define SOLAR_POTENTIAL_H     10 // direct + diffuse (W m-2)
#define REFLECTIVE_IRD_H      11 // reflected shortwave, first order (W m-2)
#define REFLECTIVE_IRD_H2     12 // reflected shortwave, second order (W m-2)
#define LONGWAVE_SURR_H       13 // longwave from the surrounding surfaces, from the viewshed caches (W m-2)
#define NET_RAD_OLD           14 // net radiation of the previous hour (W m-2), for the OHM hysteresis term
#define SOIL_MOISTURE         15 // soil water bucket (mm)
#define REFLECTIVE_IRD_SEB_H  16 // reflected shortwave used in the SEB (W m-2)
#define LONGWAVE_SKY_H        17 // Lsky (W m-2)
#define LONGWAVE_SURR_SEB_H   18 // Lsurr used in the SEB (W m-2)
#define NET_RAD_H             19 // Q* (W m-2)
#define STORAGE_H             20 // storage heat flux, OHM (W m-2)
#define LATENT_H              21 // latent heat flux QE (W m-2)
#define SENSIBLE_H            22 // sensible heat flux QH = h_c (Ts - Ta) (W m-2)
#define H_C_H                 23 // convective heat transfer coefficient (W m-2 K-1)
#define WINDSPEED_H           24 // local wind speed (m s-1)
#define HEIGHT_AGL            25 // height above the local ground (m), for the wind profile: 0 at street level, the building
                                 // height on roofs
#define GRID_POS              26 // raster cell index x + y * width of the voxel's column; must stay NUM_ATTRIBUTES - 1
#define NUM_ATTRIBUTES        27

#define SCALE_VAL 10 // heights are packed into the GPU height raster as 16-bit integers of 1/SCALE_VAL m (rounded)
// heights of the GPU height raster, read back as float (m above the height reference)
#define GRID_Z(g)  ((float)((g) >> 16) * (1.0f / SCALE_VAL))      // terrain or building (upper 16 bits)
#define GRID_ZV(g) ((float)((g) & 0xFFFFu) * (1.0f / SCALE_VAL))  // vegetation (lower 16 bits)

// Viewshed cache entry: voxel index in bits 0-22, with lsurr_ray_weights the number of rays that hit the voxel
// minus 1 in bits 23-30; negative = seen through vegetation
#define SVF_ID_MASK   0x7FFFFF
#define GRID_NO_TARGET 0xFFFFFFFFu // grid_t entry for a raster cell with no voxel
#define SVF_CNT_SHIFT 23

#define HOURS_PER_YEAR 8760     // forcing series and simulated days use 365 days
#define FORCING_ARRAY_SIZE 8784 // device forcing arrays (room for a leap year)
