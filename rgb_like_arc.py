import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# افتح ملف DEM
with rasterio.open("dem.tif") as src:
    dem = src.read(1)
    transform = src.transform
    profile = src.profile

# احسب الميل
xres, yres = transform.a, -transform.e
dzdx, dzdy = np.gradient(dem, xres, yres)
slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)

# إنشاء colormap من matplotlib
plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(slope, cmap='terrain')  # جرب 'viridis' أو 'gist_earth' كمان
plt.colorbar(label='Slope (degrees)')
plt.savefig("slope_coloreed.png", bbox_inches='tight', pad_inches=0)
plt.close()

print("✅ Slope with colormap saved as slope_colored.png")
