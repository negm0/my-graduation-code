import rasterio
import numpy as np
import os
from skimage.measure import find_contours

# تحديد المسار الصحيح للملف داخل المشروع
project_dir = r"C:\Users\mahmo\PycharmProjects\PythonProject1"
dem_file = os.path.join(project_dir, "dem.TIF")
contour_file = os.path.join(project_dir, "counterrr4.TIF")  # تغيير اسم الملف النهائي


# تحميل بيانات DEM
def load_dem(file_path):
    with rasterio.open(file_path) as dataset:
        dem_data = dataset.read(1)  # قراءة بيانات الارتفاعات
        transform = dataset.transform
        crs = dataset.crs
        return dem_data, transform, crs


# حساب خطوط الكنتور باستخدام scikit-image وحفظ قيم الارتفاعات
def generate_contour(dem_data, level_step=5000):
    min_val, max_val = np.min(dem_data), np.max(dem_data)
    levels = np.arange(min_val, max_val, level_step)

    contour_mask = np.full_like(dem_data, fill_value=np.nan, dtype=np.float32)  # جعل الخلفية NaN

    for level in levels:
        contours = find_contours(dem_data, level)
        for contour in contours:
            for x, y in contour:
                if 0 <= int(y) < contour_mask.shape[0] and 0 <= int(x) < contour_mask.shape[1]:
                    contour_mask[int(y), int(x)] = level  # تعيين قيمة الارتفاع الفعلية

    # تصحيح دوران الصورة عبر قلبها رأسيًا وأفقيًا
    contour_mask = np.flipud(np.fliplr(contour_mask))

    return contour_mask


# حفظ الكنتور كملف TIF مع تضمين قيم الارتفاعات
def save_contour_as_tif(contour_data, transform, crs, output_path):
    # تعديل `transform` لحساب الإزاحة الصحيحة بعد القلب
    new_transform = rasterio.transform.Affine(
        transform.a, transform.b, transform.c,
        transform.d, -transform.e, transform.f + (transform.e * contour_data.shape[0])
    )

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=contour_data.shape[0],
            width=contour_data.shape[1],
            count=1,
            dtype=rasterio.float32,  # تغيير نوع البيانات لحفظ قيم الارتفاعات
            crs=crs,
            transform=new_transform,
            nodata=np.nan  # تحديد NoData بـ NaN
    ) as dst:
        dst.write(contour_data, 1)


# تنفيذ الكود
if os.path.exists(dem_file):
    dem_array, dem_transform, dem_crs = load_dem(dem_file)
    contour_array = generate_contour(dem_array)
    save_contour_as_tif(contour_array, dem_transform, dem_crs, contour_file)
    print(f"✅ تم حفظ خطوط الكنتور بنجاح في: {contour_file}")
else:
    print(f"⚠️ الملف غير موجود: {dem_file}")
