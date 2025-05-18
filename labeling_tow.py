import matplotlib
matplotlib.use('Agg')

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

dem_path = r"C:\Users\mahmo\PycharmProjects\PythonProject1\dem_Clip.tif"
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype('float32')
    transform = src.transform
    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan

    height, width = dem.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x = transform.c + cols * transform.a + rows * transform.b
    y = transform.f + cols * transform.d + rows * transform.e

# إعداد الخريطة
plt.figure(figsize=(10, 8))

# مستويات الكنتور + الألوان المقابلة
levels = [12000, 15000, 18000, 21000]
colors = ['lightgreen', 'violet', 'lightblue', 'black']

# رسم كل مستوى بلونه المحدد
for level, color in zip(levels, colors):
    cs = plt.contour(x, y, dem, levels=[level], colors=[color], linewidths=1.5)
    plt.clabel(cs, fmt='%d', fontsize=8, colors='red')

# العناوين والمحاور
plt.title("Contour Map with Colored Levels")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")

# تقليل التكات على المحاور
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

# حفظ الخريطة
plt.savefig("contour_colored_labeled5.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ تم حفظ خريطة الكنتور بألوان مخصصة وأرقام: contour_colored_labeled4.png")
