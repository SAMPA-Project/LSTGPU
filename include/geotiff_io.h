#pragma once

// Reading single-band rasters and writing multi-band 32-bit float GeoTIFF rasters

#include <cstdint>
#include <string>
#include <vector>

namespace lstgpu {

/* GeoTIFF metadata for one raster: standard TIFF image tags plus the GeoTIFF-specific tags (tiepoint/pixel-scale/
   geo-keys) needed to place it in map space. */
struct TiffMeta {
    uint32_t w = 0, h = 0;
    uint16_t sampleFormat = 0, bitsPerSample = 0, samplesPerPixel = 0;
    uint16_t compression = 0, photometric = 0, planarConfig = 0;
    uint32_t rowsPerStrip = 0;

    std::vector<double> tiepoints;   // GeoTIFF tag 33922 (ModelTiepointTag)
    std::vector<double> pixelScale;  // GeoTIFF tag 33550 (ModelPixelScaleTag)
    std::vector<uint16_t> geoKeys;   // GeoTIFF tag 34735 (GeoKeyDirectoryTag)
    std::vector<double> geoDoubles;  // GeoTIFF tag 34736 (GeoDoubleParamsTag)
    std::string geoAscii;            // GeoTIFF tag 34737 (GeoAsciiParamsTag)
};

/* Placement of an output raster: upper-left corner, square pixel size, and the coordinate reference system as
   GeoTIFF keys (empty: no CRS written). */
struct GeoRef {
    double origin_x = 0.0, origin_y = 0.0; // map coordinates of the upper-left corner of the upper-left pixel
    double res = 1.0;
    std::vector<uint16_t> geoKeys;
    std::vector<double> geoDoubles;
    std::string geoAscii;
};

struct BandInfo {
    std::string name, unit;
};

/* Registers the GeoTIFF custom tags (33550/33922/34735/34736/34737) plus GDAL_METADATA (42112) and GDAL_NODATA
   (42113) with libtiff. Must be called once before any read or write. Safe to call more than once. */
void register_geotiff_tags();

/* Reads the first band of a TIFF (any 8-64 bit integer or float sample type; strips or tiles) as float. Exits the
   process with an error message if the file can't be read. */
std::vector<float> read_tiff_band(const std::string& path, TiffMeta& meta);

/* Writes `data` (w * h * bands.size() floats, row-major, bands interleaved per pixel) as a deflate-compressed 32-bit
   float GeoTIFF with NaN as no data, and the band names and units as GDAL band descriptions. Exits the process if
   the file can't be written. */
void write_tiff_bands(const std::string& path, uint32_t w, uint32_t h, const std::vector<float>& data,
                      const std::vector<BandInfo>& bands, const GeoRef& geo);

/* GeoTIFF keys for a projected CRS: the keys of `keys` (from a LAS file) with the raster type set to PixelIsArea,
   or, without keys, ProjectedCSTypeGeoKey = epsg. Fills geo.geoKeys/geoDoubles/geoAscii. */
void set_crs(GeoRef& geo, const std::vector<uint16_t>& keys, const std::vector<double>& doubles,
             const std::string& ascii, int epsg);

} // namespace lstgpu
