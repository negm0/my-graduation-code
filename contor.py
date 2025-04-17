import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

# تحديد المسار الصحيح للملف داخل المشروع
project_dir = r"C:\Users\mahmo\PycharmProjects\PythonProject1"
dem_file = os.path.join(project_dir, "dem.TIF")

# تحميل بيانات DEM
def load_dem(file_path):
    with rasterio.open(file_path) as dataset:
        dem_data = dataset.read(1)  # قراءة بيانات الارتفاعات
        transform = dataset.transform
        extent = [dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top]
    return dem_data, extent

# رسم خطوط الكنتور
def plot_contour(dem_data, extent, contour_intervals=20):
    plt.figure(figsize=(10, 6))
    plt.contour(dem_data, levels=contour_intervals, cmap="terrain", extent=extent)
    plt.colorbar(label="الارتفاع (متر)")
    plt.title("خريطة خطوط الكنتور")
    plt.xlabel("الإحداثي الشرقي")
    plt.ylabel("الإحداثي الشمالي")
    plt.show()

# تنفيذ الكود باستخدام ملف DEM الموجود في المشروع
if os.path.exists(dem_file):
    dem_data, extent = load_dem(dem_file)
    plot_contour(dem_data, extent)
else:
    print(f"⚠️ الملف غير موجود: {dem_file}")

