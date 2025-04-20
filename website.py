from flask import Flask, render_template, request, send_file
import os

from matplotlib.patches import Patch
from skimage.measure import find_contours
from werkzeug.utils import secure_filename
import tempfile


import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('Agg')  # علشان نحل مشكلة واجهة العرض

import rasterio
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def generate_contour_colored(dem_data, level_step):
    min_val, max_val = np.min(dem_data), np.max(dem_data)
    levels = np.arange(min_val, max_val, level_step)

    # خلفية بيضاء (RGB)
    contour_rgb = np.ones((dem_data.shape[0], dem_data.shape[1], 3), dtype=np.uint8) * 255

    # ألوان عشوائية للمستويات (يمكنك تحديد ألوان ثابتة لو تحب)
    colors = plt.cm.jet(np.linspace(0, 1, len(levels)))[:, :3] * 255  # أخذ ألوان من colormap وتحويلها لـ RGB

    for i, level in enumerate(levels):
        contours = find_contours(dem_data, level)
        color = colors[i].astype(np.uint8)
        for contour in contours:
            for x, y in contour:
                x, y = int(x), int(y)
                if 0 <= y < dem_data.shape[0] and 0 <= x < dem_data.shape[1]:
                    contour_rgb[y, x] = color

    # قلب الاتجاه لو لازم
    contour_rgb = np.flipud(contour_rgb)
    return contour_rgb

# توليد خطوط الكنتور
def generate_contour(dem_data, level_step):
    min_val, max_val = np.min(dem_data), np.max(dem_data)
    levels = np.arange(min_val, max_val, level_step)

    print("levels count: " + str(len(levels)))
    contour_mask = np.full((dem_data.shape[0], dem_data.shape[1], 3), fill_value=[255,255,255], dtype=np.uint8)
    for level in levels:
        # TODO: اتاكد ان الالوان واضحة على خلفية بيضة
        random_rgb = np.random.randint(0, 256, size=3).tolist()
        contours = find_contours(dem_data, level)
        for contour in contours:
            for x, y in contour:
                if 0 <= int(y) < contour_mask.shape[0] and 0 <= int(x) < contour_mask.shape[1]:
                    contour_mask[int(y), int(x)] = random_rgb

    contour_mask = np.flipud(contour_mask)
    return contour_mask


def save_rgb_contour_tif(contour_mask, transform, crs, output_path):
    # Flip transform if needed (because of np.flipud)
    new_transform = rasterio.transform.Affine(
        transform.a, transform.b, transform.c,
        transform.d, -transform.e, transform.f + (transform.e * contour_mask.shape[0])
    )

    # Rearrange from (H, W, 3) → (3, H, W)
    contour_mask = contour_mask.transpose((2, 0, 1))

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=contour_mask.shape[1],
        width=contour_mask.shape[2],
        count=3,  # RGB = 3 bands
        dtype=rasterio.uint8,
        crs=crs,
        transform=new_transform
    ) as dst:
        dst.write(contour_mask)

def save_contour_as_tif(contour_data, transform, crs, output_path):
    new_transform = rasterio.transform.Affine(
        transform.a, transform.b, transform.c,
        transform.d, -transform.e, transform.f + (transform.e * contour_data.shape[0])
    )

    if contour_data.ndim == 3 and contour_data.shape[2] == 3:
        contour_data = contour_data.transpose((2, 0, 1))
        count = 3
    else:
        count = 1

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=contour_data.shape[1] if count == 3 else contour_data.shape[0],
        width=contour_data.shape[2] if count == 3 else contour_data.shape[1],
        count=count,
        dtype=rasterio.uint8,
        crs=crs,
        transform=new_transform
    ) as dst:
        dst.write(contour_data)

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['dem_file']
        interval = int(request.form['interval'])

        if file and interval > 0:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            with rasterio.open(filepath) as dataset:
                dem_data = dataset.read(1)
                transform = dataset.transform
                crs = dataset.crs

            contour_array = generate_contour(dem_data, interval)

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contour_result.TIF')
            save_rgb_contour_tif(contour_array, transform, crs, output_path)

            return send_file(output_path, as_attachment=True)

    return render_template('index.html')

@app.route("/slop", methods=['POST'])
def saveSlop():
    file = request.files['dem_file']
    with rasterio.open(file) as src:
        dem = src.read(1)
        transform = src.transform
        profile = src.profile

    xres, yres = transform.a, -transform.e

    dzdx, dzdy = np.gradient(dem, xres, yres)
    slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)) * (180 / np.pi)

    profile.update(dtype=rasterio.float32, count=1)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'slop_result.TIF')
    # احفظ النتيجة في ملف جديد
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope.astype(np.float32), 1)

    return send_file(output_path, as_attachment=True)
@app.route("/Aspect", methods=['POST'])
def generate_aspect_map():
    # قراءة ملف الـ DEM
    file = request.files['dem_file']  # استلام الملف من الفورم

    with rasterio.open(file) as src:
        dem = src.read(1)
        transform = src.transform
        pixel_size_x = src.res[0]
        pixel_size_y = src.res[1]

    # حساب المشتقات
    dy, dx = np.gradient(dem, pixel_size_y, pixel_size_x)

    # حساب الـ Aspect
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360 + aspect_deg, aspect_deg)

    # تحديد المناطق المسطحة (Flat)
    flat_mask = (dx == 0) & (dy == 0)
    aspect_deg[flat_mask] = -1

    # تصنيف الاتجاهات
    aspect_classes = np.full_like(aspect_deg, 0)
    aspect_classes[(aspect_deg == -1)] = 0  # Flat
    aspect_classes[(aspect_deg >= 0) & (aspect_deg < 22.5)] = 1
    aspect_classes[(aspect_deg >= 22.5) & (aspect_deg < 67.5)] = 2
    aspect_classes[(aspect_deg >= 67.5) & (aspect_deg < 112.5)] = 3
    aspect_classes[(aspect_deg >= 112.5) & (aspect_deg < 157.5)] = 4
    aspect_classes[(aspect_deg >= 157.5) & (aspect_deg < 202.5)] = 5
    aspect_classes[(aspect_deg >= 202.5) & (aspect_deg < 247.5)] = 6
    aspect_classes[(aspect_deg >= 247.5) & (aspect_deg < 292.5)] = 7
    aspect_classes[(aspect_deg >= 292.5) & (aspect_deg < 337.5)] = 8
    aspect_classes[(aspect_deg >= 337.5) & (aspect_deg <= 360)] = 1

    # الألوان لكل اتجاه
    colors = [
        "#d9d9d9",  # Flat
        "#ff0000",  # North
        "#ff9900",  # NE
        "#ffff00",  # East
        "#99cc00",  # SE
        "#339966",  # South
        "#33cccc",  # SW
        "#3366cc",  # West
        "#9900cc",  # NW
    ]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, 9.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # رسم الخريطة
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(aspect_classes, cmap=cmap, norm=norm)
    ax.set_title("Aspect Directions", fontsize=14)
    ax.axis('off')

    legend_labels = [
        "Flat", "North", "Northeast", "East", "Southeast",
        "South", "Southwest", "West", "Northwest"
    ]
    legend_patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # اسم الصورة على نفس مسار الـ DEM
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aspect_result.PNG')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close()

    print(f"✅ تم حفظ خريطة Aspect المصنفة هنا: {output_path}")
    # احفظ النتيجة في ملف جديد
    #with rasterio.open(output_path, "w", **profile) as dst:
       # dst.write(slope.astype(np.float32), 1)
#
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

