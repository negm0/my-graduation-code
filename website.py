from flask import Flask, render_template, request, send_file
import os
import numpy as np
import rasterio
from skimage.measure import find_contours
from werkzeug.utils import secure_filename
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

if __name__ == '__main__':
    app.run(debug=True)
