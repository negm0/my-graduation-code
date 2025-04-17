import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

# تحديد المسار الصحيح للملف داخل المشروع
project_dir = r"C:\Users\mahmo\PycharmProjects\PythonProject1"
dem_file = os.path.join(project_dir, "dem.TIF")

# تحميل بيانات DEM
def load_dem(file_path):
    with rasterio.open(file_path) as dataset:
        dem_data = dataset.read(1)  # قراءة بيانات الارتفاعات
        dem_extent = [dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top]
    return dem_data, dem_extent

# رسم خطوط الكنتور بفاصل محدد
def plot_contour(dem_data_array, dem_extent, contour_interval=10):
    min_elev = np.nanmin(dem_data_array)  # أقل ارتفاع
    max_elev = np.nanmax(dem_data_array)  # أعلى ارتفاع
    contour_levels = np.arange(min_elev, max_elev, contour_interval)  # تحديد مستويات الكنتور

    plt.figure(figsize=(10, 6))
    contour = plt.contour(dem_data_array, levels=contour_levels, cmap="terrain", extent=dem_extent)
    plt.colorbar(contour, label="الارتفاع (متر)")
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.0f")  # إضافة تسميات للكنتور
    plt.title("خريطة خطوط الكنتور")
    plt.xlabel("الإحداثي الشرقي")
    plt.ylabel("الإحداثي الشمالي")
    plt.show()

# تنفيذ الكود باستخدام ملف DEM الموجود في المشروع
if os.path.exists(dem_file):
    dem_array, extent_values = load_dem(dem_file)
    plot_contour(dem_array, extent_values, contour_interval=50000)  # يمكنك تغيير الفاصل الكنتوري هنا
else:
    print(f"⚠️ الملف غير موجود: {dem_file}")
