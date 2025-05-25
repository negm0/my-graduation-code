import rasterio
import numpy as np
from skimage import measure
from shapely.geometry import LineString, mapping
import fiona
from fiona.crs import from_epsg

# مسارات الملفات
input_tif = r"C:\Users\mahmo\PycharmProjects\PythonProject1\dem_Clip.tif   "   # ملف الإدخال
output_shp = "contours_try1_to_shp.shp"   # ملف الإخراج

# فاصل خطوط الكنتور
interval = 3000

# افتح ملف الراستر
with rasterio.open(input_tif) as src:
    band = src.read(1)
    transform = src.transform
    nodata = src.nodata
    crs = src.crs

    # استبعد NoData
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)

    # حدد مستويات الكنتور
    min_val = np.nanmin(band)
    max_val = np.nanmax(band)
    levels = np.arange(min_val, max_val, interval)

    # جهز هيكل ملف الـ Shapefile
    schema = {
        'geometry': 'LineString',
        'properties': {'elev': 'float'},
    }

    with fiona.open(output_shp, 'w', driver='ESRI Shapefile',
                    crs=crs.to_dict(), schema=schema) as shp:

        # استخراج الكونتور لكل مستوى
        for level in levels:
            contours = measure.find_contours(band, level=level)

            for contour in contours:
                # تحويل الإحداثيات من Pixel إلى Geo
                coords = [rasterio.transform.xy(transform, y, x) for y, x in contour]
                line = LineString(coords)

                shp.write({
                    'geometry': mapping(line),
                    'properties': {'elev': float(level)},
                })

print("✅ تم استخراج خطوط الكنتور بدون استخدام GDAL")
