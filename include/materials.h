#pragma once

/* Material class (row of the material table, model_params.h) of every top voxel. Wall voxels are "wall" (11) and
   the ground under canopies is "soil" (2). */

#include <string>

#include "surface_raster.h"

namespace lstgpu {

/* Top voxels get the class of the material raster pixel under the cell centre, where `raster_path` is given, covers
   the cell and holds a class 0..14; all other top voxels get one from the height rasters: building 14 (average
   roof), vegetation 5 (shrubs and trees), ground 0 (average ground). The raster is placed by its GeoTIFF tags or a
   world file next to it (.tfw), in the coordinate system of the input rasters. Returns the number of top voxels with
   a class from the raster. */
int assign_materials(SurfaceModel& sm, const std::string& raster_path);

} // namespace lstgpu
