import rasterio
import numpy as np

# افتح ملف DEM واقرأ البيانات
with rasterio.open("dem.tif") as src:
    dem = src.read(1)  # قراءة بيانات الارتفاعات
    transform = src.transform  # الحصول على معلومات التحويل الجغرافي
    profile = src.profile  # حفظ معلومات الملف لاستخدامها لاحقًا

# احسب دقة البيكسل
xres, yres = transform.a, -transform.e

# احسب الميل باستخدام التدرج
dzdx, dzdy = np.gradient(dem, xres, yres)
slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)

# تحديث معلومات الملف للحفظ
profile.update(dtype=rasterio.float32, count=1)

# احفظ النتيجة في ملف جديد
with rasterio.open("slope_test_ali.tif", "w", **profile) as dst:
    dst.write(slope.astype(np.float32), 1)

print("✅ Slope calculation completed! File saved as slope.tif")
