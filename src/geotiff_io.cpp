#include "geotiff_io.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include <tiffio.h>

namespace lstgpu {

namespace {

/* libtiff's classic API doesn't know about GeoTIFF tags out of the box; this table teaches it their type/count so
   TIFFGetField/TIFFSetField work for them. See the GeoTIFF spec for tag numbers: 33550 ModelPixelScaleTag,
   33922 ModelTiepointTag, 34735 GeoKeyDirectoryTag, 34736 GeoDoubleParamsTag, 34737 GeoAsciiParamsTag.
   42112 (GDAL_METADATA) and 42113 (GDAL_NODATA) are GDAL's conventions, not part of the GeoTIFF spec proper. */
const TIFFFieldInfo kGeotiffFieldInfo[] = {
    { 33550, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelPixelScaleTag" },
    { 33922, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTiepointTag" },
    { 34735, -1, -1, TIFF_SHORT,  FIELD_CUSTOM, 1, 1, (char*)"GeoKeyDirectoryTag" },
    { 34736, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"GeoDoubleParamsTag" },
    { 34737, -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GeoAsciiParamsTag" },
    { 42112, -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GDAL_METADATA" },
    { 42113, -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GDAL_NODATA" },
};

TIFFExtendProc g_parentExtender = nullptr;

void geotiff_tag_extender(TIFF* tif) {
    TIFFMergeFieldInfo(tif, kGeotiffFieldInfo, sizeof(kGeotiffFieldInfo) / sizeof(kGeotiffFieldInfo[0]));
    if (g_parentExtender) g_parentExtender(tif);
}

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(EXIT_FAILURE);
}

// Sample k of a raw buffer as float
float sample_as_float(const unsigned char* buf, size_t k, uint16_t fmt, uint16_t bps) {
    if (fmt == SAMPLEFORMAT_IEEEFP) {
        if (bps == 32) return ((const float*)buf)[k];
        if (bps == 64) return (float)((const double*)buf)[k];
    }
    else if (fmt == SAMPLEFORMAT_INT) {
        if (bps == 8) return (float)((const int8_t*)buf)[k];
        if (bps == 16) return (float)((const int16_t*)buf)[k];
        if (bps == 32) return (float)((const int32_t*)buf)[k];
    }
    else { // unsigned (also the default without a SampleFormat tag)
        if (bps == 8) return (float)buf[k];
        if (bps == 16) return (float)((const uint16_t*)buf)[k];
        if (bps == 32) return (float)((const uint32_t*)buf)[k];
    }
    return 0.0f;
}

std::string xml_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '&') o += "&amp;";
        else if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else o += c;
    }
    return o;
}

} // namespace

void register_geotiff_tags() {
    static bool registered = false;
    if (!registered) {
        g_parentExtender = TIFFSetTagExtender(geotiff_tag_extender);
        registered = true;
    }
}

std::vector<float> read_tiff_band(const std::string& path, TiffMeta& meta) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) fail("Cannot open " + path);

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &meta.w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &meta.h);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &meta.sampleFormat);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &meta.bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &meta.samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &meta.compression);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &meta.photometric);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &meta.planarConfig);
    if (!TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &meta.rowsPerStrip))
        meta.rowsPerStrip = meta.h;

    /* GeoTIFF tags: libtiff hands back pointers into its own internal buffers, so copy everything out into owned
       vectors immediately. */
    {
        double* tiepoints = nullptr; uint16_t tiepointCount = 0;
        double* pixelScale = nullptr; uint16_t pixelScaleCount = 0;
        uint16_t* geoKeys = nullptr; uint16_t geoKeyCount = 0;
        double* geoDoubles = nullptr; uint16_t geoDoubleCount = 0;
        char* geoAscii = nullptr; uint16_t geoAsciiCount = 0;

        TIFFGetField(tif, 33922, &tiepointCount, &tiepoints);
        TIFFGetField(tif, 33550, &pixelScaleCount, &pixelScale);
        TIFFGetField(tif, 34735, &geoKeyCount, &geoKeys);
        TIFFGetField(tif, 34736, &geoDoubleCount, &geoDoubles);
        TIFFGetField(tif, 34737, &geoAsciiCount, &geoAscii);

        if (tiepoints && tiepointCount) meta.tiepoints.assign(tiepoints, tiepoints + tiepointCount);
        if (pixelScale && pixelScaleCount) meta.pixelScale.assign(pixelScale, pixelScale + pixelScaleCount);
        if (geoKeys && geoKeyCount) meta.geoKeys.assign(geoKeys, geoKeys + geoKeyCount);
        if (geoDoubles && geoDoubleCount) meta.geoDoubles.assign(geoDoubles, geoDoubles + geoDoubleCount);
        if (geoAscii && geoAsciiCount) meta.geoAscii.assign(geoAscii, geoAscii + geoAsciiCount);
    }

    const uint16_t bps = meta.bitsPerSample, fmt = meta.sampleFormat;
    if (bps != 8 && bps != 16 && bps != 32 && bps != 64) {
        TIFFClose(tif);
        fail("Unsupported bits per sample " + std::to_string(bps) + " (" + path + ")");
    }
    // samples per pixel in one strip/tile row: all of them if interleaved, one per plane otherwise
    const size_t spp = (meta.planarConfig == PLANARCONFIG_SEPARATE) ? 1 : std::max<uint16_t>(meta.samplesPerPixel, 1);

    std::vector<float> data((size_t)meta.w * meta.h);

    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0, tileH = 0;
        if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW) || !TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH)) {
            TIFFClose(tif);
            fail("Tiled TIFF missing tile dimensions: " + path);
        }
        std::vector<unsigned char> tileBuf(TIFFTileSize(tif));
        for (uint32_t ty = 0; ty < meta.h; ty += tileH)
            for (uint32_t tx = 0; tx < meta.w; tx += tileW) {
                if (TIFFReadTile(tif, tileBuf.data(), tx, ty, 0, 0) < 0) {
                    TIFFClose(tif);
                    fail("Error reading tile at (" + std::to_string(tx) + ", " + std::to_string(ty) + ") of " + path);
                }
                const uint32_t copyW = std::min<uint32_t>(tileW, meta.w - tx);
                const uint32_t copyH = std::min<uint32_t>(tileH, meta.h - ty);
                for (uint32_t r = 0; r < copyH; ++r)
                    for (uint32_t c = 0; c < copyW; ++c)
                        data[(size_t)(ty + r) * meta.w + tx + c] = sample_as_float(tileBuf.data(), ((size_t)r * tileW + c) * spp, fmt, bps);
            }
    }
    else {
        std::vector<unsigned char> line(TIFFScanlineSize(tif));
        for (uint32_t row = 0; row < meta.h; ++row) {
            if (TIFFReadScanline(tif, line.data(), row, 0) < 0) {
                TIFFClose(tif);
                fail("Error reading row " + std::to_string(row) + " of " + path);
            }
            for (uint32_t c = 0; c < meta.w; ++c)
                data[(size_t)row * meta.w + c] = sample_as_float(line.data(), (size_t)c * spp, fmt, bps);
        }
    }

    TIFFClose(tif);
    return data;
}

void write_tiff_bands(const std::string& path, uint32_t w, uint32_t h, const std::vector<float>& data,
                      const std::vector<BandInfo>& bands, const GeoRef& geo) {
    const uint16_t nb = (uint16_t)bands.size();
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    if (!tif) fail("Cannot create " + path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, nb);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    if (nb > 1) {
        std::vector<uint16_t> extra(nb - 1, EXTRASAMPLE_UNSPECIFIED);
        TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, (uint16_t)(nb - 1), extra.data());
    }
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
    TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
    TIFFSetField(tif, TIFFTAG_ZIPQUALITY, 6);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));

    double pixelScale[3] = { geo.res, geo.res, 0.0 };
    double tiepoint[6] = { 0.0, 0.0, 0.0, geo.origin_x, geo.origin_y, 0.0 };
    TIFFSetField(tif, 33550, (uint16_t)3, pixelScale);
    TIFFSetField(tif, 33922, (uint16_t)6, tiepoint);
    if (!geo.geoKeys.empty())
        TIFFSetField(tif, 34735, (uint16_t)geo.geoKeys.size(), geo.geoKeys.data());
    if (!geo.geoDoubles.empty())
        TIFFSetField(tif, 34736, (uint16_t)geo.geoDoubles.size(), geo.geoDoubles.data());
    if (!geo.geoAscii.empty())
        TIFFSetField(tif, 34737, geo.geoAscii.c_str());

    // band names and units, read by GDAL (QGIS) as band descriptions
    std::ostringstream md;
    md << "<GDALMetadata>\n";
    for (uint16_t b = 0; b < nb; ++b) {
        md << "  <Item name=\"DESCRIPTION\" sample=\"" << b << "\" role=\"description\">" << xml_escape(bands[b].name) << "</Item>\n";
        if (!bands[b].unit.empty())
            md << "  <Item name=\"UNITTYPE\" sample=\"" << b << "\" role=\"unittype\">" << xml_escape(bands[b].unit) << "</Item>\n";
    }
    md << "</GDALMetadata>";
    TIFFSetField(tif, 42112, md.str().c_str());
    TIFFSetField(tif, 42113, "nan");

    for (uint32_t row = 0; row < h; ++row) {
        if (TIFFWriteScanline(tif, (void*)(data.data() + (size_t)row * w * nb), row, 0) < 0) {
            TIFFClose(tif);
            fail("Error writing row " + std::to_string(row) + " of " + path);
        }
    }

    TIFFClose(tif);
}

void set_crs(GeoRef& geo, const std::vector<uint16_t>& keys, const std::vector<double>& doubles,
             const std::string& ascii, int epsg) {
    geo.geoKeys.clear();
    geo.geoDoubles.clear();
    geo.geoAscii.clear();
    struct Key { uint16_t id, loc, count, value; };
    std::vector<Key> entries;
    if (keys.size() >= 4) {
        const size_t n = std::min<size_t>(keys[3], (keys.size() - 4) / 4);
        for (size_t k = 0; k < n; ++k) {
            Key e{ keys[4 + 4 * k], keys[5 + 4 * k], keys[6 + 4 * k], keys[7 + 4 * k] };
            if (e.id != 1025) entries.push_back(e); // GTRasterTypeGeoKey, set below
        }
        geo.geoDoubles = doubles;
        geo.geoAscii = ascii;
    }
    else if (epsg > 0) {
        entries.push_back({ 1024, 0, 1, 1 });              // GTModelTypeGeoKey: projected
        entries.push_back({ 3072, 0, 1, (uint16_t)epsg }); // ProjectedCSTypeGeoKey
    }
    else return;
    entries.push_back({ 1025, 0, 1, 1 }); // GTRasterTypeGeoKey: PixelIsArea
    std::sort(entries.begin(), entries.end(), [](const Key& a, const Key& b) { return a.id < b.id; });

    geo.geoKeys = { 1, 1, 0, (uint16_t)entries.size() };
    for (const auto& e : entries) {
        geo.geoKeys.push_back(e.id);
        geo.geoKeys.push_back(e.loc);
        geo.geoKeys.push_back(e.count);
        geo.geoKeys.push_back(e.value);
    }
}

} // namespace lstgpu
