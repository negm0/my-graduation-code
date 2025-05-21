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


def generate_colored_contour_png(dem_data, transform, output_path, interval):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # تحضير الإحداثيات
    height, width = dem_data.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x = transform.c + cols * transform.a + rows * transform.b
    y = transform.f + cols * transform.d + rows * transform.e

    # توليد المستويات
    min_val, max_val = np.nanmin(dem_data), np.nanmax(dem_data)
    levels = np.arange(min_val, max_val, interval)

    # Colormap تلقائي أو ثابت
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(levels)) for i in range(len(levels))]

    # رسم الكونتور
    plt.figure(figsize=(10, 8))
    for i, level in enumerate(levels):
        cs = plt.contour(x, y, dem_data, levels=[level], colors=[colors[i]], linewidths=1.5)
        plt.clabel(cs, fmt='%d', fontsize=8, colors='black')

    plt.title("Colored Contour Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    # حفظ الصورة
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

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
                dem_data = dataset.read(1).astype('float32')
                transform = dataset.transform
                if dataset.nodata is not None:
                    dem_data[dem_data == dataset.nodata] = np.nan

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contour_result.PNG')
            generate_colored_contour_png(dem_data, transform, output_path, interval)

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

    # --- إعداد colormap مخصص للعرض (مشابه لـ ArcGIS) ---
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#00ff00',  # 0-5
              '#ffff00',  # 5-15
              '#ffa500',  # 15-30
              '#ff0000',  # 30-45
              '#800000']  # 45-90
    bounds = [0, 5, 15, 30, 45, 90]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- رسم الخريطة ---
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(slope, cmap=cmap, norm=norm)
    cbar = plt.colorbar(img, ticks=bounds, ax=ax)
    cbar.set_label('Slope (degrees)')
    ax.set_title('Slope Map - ArcGIS Style Colormap')
    ax.axis('off')

    # --- حفظ الصورة ---
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'slope_result_arcgis_style.PNG')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

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

