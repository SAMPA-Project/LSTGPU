# LSTGPU

This implementation is supplementary material for the paper *High-resolution energy balance based simulation of land surface temperature using remote sensing data* by Niko Lukac, Andreas Holzinger and Marko Bizjak (University of Maribor; BOKU University, Vienna), under review.

**Version: 1.0**

## Description

GPU-accelerated simulation of the land surface temperature (LST) at 1 m over a 3D voxel-based surface model derived from LiDAR point cloud. For every surface voxel the full surface energy balance (SEB) is solved hourly:

- **shortwave:** direct irradiance with shadowing, anisotropic diffuse irradiance (Perez model with the visible parts of the sky, horizon band and circumsolar region), and two orders of reflection from the surrounding surfaces;
- **longwave:** from the sky (Brutsaert clear-sky emissivity with a cloud correction; a canopy in the line of sight emits at air temperature) and from the surrounding surfaces at their simulated LST;
- **sky view factor:** the radiative (cosine-weighted) view factor to the sky, so that the sky and the surrounding surfaces share the view consistently;
- **storage heat flux:** objective hysteresis model (OHM) per material, with a calibrated seasonal term;
- **sensible heat:** forced and natural convection, with the local wind from a power-law profile at the height above the local ground and dynamic wind sheltering;
- **latent heat:** Priestley-Taylor evaporation of pervious surfaces, limited by a soil water bucket;
- **vegetation:** LAI-driven Beer-Lambert attenuation of radiation and wind through tree canopies, which also gives the LST of the ground under a canopy.

## Building

Requires a CUDA-capable GPU and toolkit (nvcc), CMake >= 3.18, a C++17 compiler and [libtiff](https://libtiff.gitlab.io/libtiff/). With [vcpkg](https://vcpkg.io/), for example:

```bash
vcpkg install tiff
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

## Running

```
LSTGPU --surface <prefix> --meteo <dir> --out <dir> --lon <deg> --lat <deg> --timezone <h> [options]
```

`--help` lists every option. The ones that matter for a run are `--days a b` (the simulated days of the year, counted from 0), `--spinup n` (days simulated before them and not written, so that the storage heat flux and the soil water are in equilibrium), `--materials` (the material class of every cell), `--window` (write only a part of the area, for example a study area without its boundary) and `--params` (model constants other than the calibrated defaults).

## Inputs

### Surface (`--surface <prefix>`)

Six single-band 32-bit float GeoTIFF rasters of one grid, with square cells and a CRS, named by a common prefix (`--surface inputs/surface/maribor` reads `inputs/surface/maribor_ground.tif` and so on):

| File | Contents |
|---|---|
| `<prefix>_ground.tif` | absolute elevation of the ground (m), 0 or NaN where the cell has no ground |
| `<prefix>_building.tif` | absolute elevation of the building surface (m), 0 or NaN where the cell has no building |
| `<prefix>_vegetation.tif` | absolute elevation of the top of the tree canopy (m), 0 or NaN where the cell has no canopy |
| `<prefix>_normal_x.tif`, `<prefix>_normal_y.tif`, `<prefix>_normal_z.tif` | the surface normal of the cell, z upwards and y northwards; all three 0 for a horizontal surface |

### Material classes (`--materials <tif>`)

A raster of the material class of every cell (0 to 14, the rows of the material table in `params/default_params.txt`), placed by its GeoTIFF tags or a world file (`.tfw`) next to it. Cells the raster does not cover, and values outside 0 to 14, fall back to the classes of the height rasters: building 14 (average roof), canopy 5 (trees and shrubs), ground 0 (average ground). Wall voxels are always class 11 and the ground under a canopy always class 2.

### Meteorological data (`--meteo <dir>`)

Nine text files with one value per line, in the fixed clock of `--timezone`, starting at 1 January 00:00:

| File | Contents |
|---|---|
| `direct.txt`, `diffuse.txt` | direct and diffuse horizontal irradiance (W m-2) |
| `temperature.txt` | air temperature (°C) |
| `humidity.txt` | relative humidity (%) |
| `windspeed.txt` | wind speed (m s-1) |
| `winddir.txt` | direction the wind blows from (degrees from north, clockwise) |
| `rainfall.txt` | precipitation (mm per hour) |
| `cloudiness.txt` | total cloud cover (0 to 1), used at night and at low irradiance |
| `lai.txt` | leaf area index of the tree canopy, one value per day of the year (365) |

The hourly series have 8760 values, or 8784 for a leap year given with `--year`. `--lai` reads the LAI from another file.

## Output

In `--out`, one raster per simulated hour, `lst_dDDD_hHH.tif` (DDD = day of the year from 0, HH = hour): the land surface temperature (°C) as a 32-bit float GeoTIFF at 1 m, NaN where there is no surface, in the CRS of the ground raster (or `--epsg`). 

## Example

The input data of the paper's simulation domain in central Maribor, Slovenia (15.6408° E, 46.5606° N), are in the paper's data record on Zenodo (DOI: to be added), in `inputs.zip`: the 683 m × 619 m study area with a 100 m boundary on all sides, 885 × 821 cells at 1 m in EPSG:3794 (Slovenia 1996 / D96/TM), 1,248,599 surface voxels. The record's README describes the data, their sources and licences.

```bash
./build/LSTGPU --surface inputs/surface/maribor --meteo inputs/meteo/2025 --lai inputs/lai/lai_2025.txt \
  --materials inputs/materials/material_map_1m.tif --out example_out --lon 15.6408 --lat 46.5606 --timezone 1 \
  --days 171 172 --spinup 1 --year 2025 \
  --window 548786 157779.15625 549469 158398.15625
```

This writes 24 hourly LST rasters of 683 × 619 cells, in about 10 s on an RTX 4090. The viewshed caches of the 1,248,599 voxels need 12.06 GiB of GPU memory, which is compacted to 5.87 GiB after the ray casting, so a GPU of 16 GB or more is needed.

## Model constants

The defaults are the values calibrated against 220 clear-sky Landsat 8/9 LST scenes of central Maribor (2013-2025): the convection scaling, the seasonal storage term, and factors on the OHM coefficient a1 of the impervious materials and on the Priestley-Taylor coefficient of the pervious ones. Classes 0 and 14 are area-weighted means of the calibrated classes, for runs without a material map. `params/default_params.txt` lists every constant with its meaning; copy it, change values and pass it with `--params` (keys that are left out keep their defaults).

## Source layout

| Path | Contents |
|---|---|
| `src/main.cpp` | command line, run settings, timing |
| `params/default_params.txt` | every model constant with its meaning, at the calibrated values |
| `src/surface_raster.cpp` | the surface rasters, and the voxels with their slope, aspect and material |
| `src/materials.cpp`, `src/meteo.cpp`, `src/model_params.cpp` | material map, meteorological data, model constants |
| `src/geotiff_io.cpp`, `src/output_writer.cpp` | GeoTIFF input and output |
| `src/sky_bins.cpp` | solid angles of the hemisphere bins, circumsolar region of an hour |
| `src/cuda/lst_pipeline.cu` | the GPU pipeline: ray casting once, then the hourly kernels |
| `src/cuda/device_globals.cu`, `src/cuda/device_upload.cu` | the device globals, and the upload of the model and the data |
| `src/cuda/kernels/` | one kernel per file: `viewshed_kernels.cu` (horizon buffers, sky view, viewshed caches), `irradiance_kernel.cu` (direct and diffuse shortwave), `reflection_kernels.cu` (reflected shortwave and longwave from the surroundings), `wind_kernel.cu` (wind sheltering), `seb_kernel.cu` (energy balance and LST) |
| `include/cuda/physics/` | the physics of the kernels, one process per header: `sky_longwave.cuh`, `convection.cuh`, `latent_heat.cuh`, `storage_heat.cuh`, `wind_profile.cuh`, `canopy.cuh`, `perez_diffuse.cuh`, `surface_geometry.cuh` |
| `include/cuda/horizon_buffer.cuh` | the quantised horizon buffers |

## Third-party code

- the PSA solar position algorithm (Blanco-Muriel et al., Solar Energy 70(5), 431–441, 2001), `include/sun_position.h`

## Acknowledgement

The authors acknowledge financial support from the Slovenian Research and Innovation Agency (Research Project No. J7-50095, and Research Core Funding No. P2-0041). This research was supported in part by the Austrian Science Fund (FWF), Project 10.55776/P32554.

## Licence

The code is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (`LICENSE`): you may use, modify and share it for any noncommercial purpose, including research and teaching, provided you keep the `Required Notice` line of `LICENSE` and cite the paper above. Commercial use needs the authors' permission. The input data of the example are not part of this repository; they keep the licences stated in the data record.
