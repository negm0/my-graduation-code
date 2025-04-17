import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

# افتح ملف DEM واقرأ البيانات
with rasterio.open("dem.tif") as src:
    dem = src.read(1)
    transform = src.transform
    profile = src.profile

# احسب دقة البيكسل
xres, yres = transform.a, -transform.e

# احسب الميل
dzdx, dzdy = np.gradient(dem, xres, yres)
slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)

# 🔥 تطبيق Colormap على الميل (مثلاً colormap 'terrain')
norm_slope = (slope - np.min(slope)) / (np.max(slope) - np.min(slope))  # Normalization to 0–1
colored_slope = cm.terrain(norm_slope)[:, :, :3]  # RGB only (drop alpha)

# تحويله من float إلى uint8
colored_slope_uint8 = (colored_slope * 255).astype(np.uint8)

# احفظ كصورة ملونة
Image.fromarray(colored_slope_uint8).save("slope_colored.png")

# ✅ احفظ الـ slope كـ TIF عادي لو عايز تستخدمه كـ Data
profile.update(dtype=rasterio.float32, count=1)
with rasterio.open("slope_test_ali.tif", "w", **profile) as dst:
    dst.write(slope.astype(np.float32), 1)

print("✅ Slope calculation + coloring completed! PNG & TIF saved.")
