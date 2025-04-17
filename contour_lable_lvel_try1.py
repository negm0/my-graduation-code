import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import rasterio

# افتح ملف DEM بصيغة TIF
with rasterio.open("dem.tif") as src:
    dem = src.read(1)  # بيانات الارتفاع
    transform = src.transform

# تحديد مستويات الكنتور
min_val, max_val = np.min(dem), np.max(dem)
level_step = 6000
levels = np.arange(min_val, max_val, level_step)

# Colormap
cmap = plt.get_cmap('terrain')

# رسم الصورة
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(dem, cmap='gray', origin='upper')

# رسم الكنتور مع التلوين والليبل
for i, level in enumerate(levels):
    contours = find_contours(dem, level)
    color = cmap(i / len(levels))
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
        if len(contour) > 0:
            mid_idx = len(contour) // 2
            y, x = contour[mid_idx]
            ax.text(x, y, f"{int(level)}", color='black', fontsize=8,
                    ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

ax.set_title("Contour Map with Colored Levels and Labels")
ax.axis('off')
plt.tight_layout()

# حفظ الناتج كصورة PNG
plt.savefig("contour_map.png", dpi=300)
print("✅ تم حفظ خريطة الكنتور كصورة: contour_lable_map.png")

plt.close()
