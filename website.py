from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
import cohere
import json

from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import Affine

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Cohere client - Replace with your API key
COHERE_API_KEY = "Tf9sHm2wSL2zlxNHnEuk0hzY7r6vRRJxVNsjJ12X"
co = cohere.Client(COHERE_API_KEY)

# Chat history dictionary to keep track of conversations
chat_histories = {}


# Generate colored contour map
def generate_colored_contour_png(dem_data, transform, output_path, interval):
    from matplotlib.ticker import MaxNLocator

    # Prepare coordinates
    height, width = dem_data.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x = transform.c + cols * transform.a + rows * transform.b
    y = transform.f + cols * transform.d + rows * transform.e

    # Generate levels
    min_val, max_val = np.nanmin(dem_data), np.nanmax(dem_data)
    levels = np.arange(min_val, max_val, interval)

    # Auto colormap
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(levels)) for i in range(len(levels))]

    # Draw contours
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

    # Save image
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


# Save contour data as GeoTIFF
import rasterio
from skimage import measure
from shapely.geometry import LineString, mapping
import fiona
from fiona.crs import from_epsg
from pyproj import CRS
import numpy as np

def save_contour_shapefile(dem_data, transform, crs, output_path, interval):
    if np.isnan(dem_data).all():
        raise ValueError("DEM contains only NaN values.")

    min_val = np.nanmin(dem_data)
    max_val = np.nanmax(dem_data)
    levels = np.arange(min_val, max_val, interval)

    schema = {
        'geometry': 'LineString',
        'properties': {'elev': 'float'}
    }

    epsg_code = None
    crs_for_fiona = None
    if crs:
        epsg_code = crs.to_epsg()
        if epsg_code:
            crs_for_fiona = from_epsg(epsg_code)
        else:
            crs_for_fiona = {'init': 'epsg:4326'}  # fallback or you can do from_string(crs.to_wkt())

    with fiona.open(output_path, 'w',
                    driver='ESRI Shapefile',
                    crs=crs_for_fiona,
                    schema=schema) as shp:
        for level in levels:
            contours = measure.find_contours(dem_data, level=level)
            for contour in contours:
                # استخدم rasterio.transform.xy لتحويل الإحداثيات بدقة
                coords = [rasterio.transform.xy(transform, row, col) for row, col in contour]
                line = LineString(coords)
                if line.is_valid:
                    shp.write({
                        'geometry': mapping(line),
                        'properties': {'elev': float(level)}
                    })

    # اكتب ملف .prj يدويا
    if epsg_code:
        prj_path = output_path.replace('.shp', '.prj')
        with open(prj_path, 'w') as prj_file:
            prj_file.write(CRS.from_epsg(epsg_code).to_wkt())








# Main route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Contour route
@app.route('/contour', methods=['POST'])
def generate_contour():
    file = request.files['dem_file']
    interval = int(request.form['interval'])
    output_format = request.form.get('output_format', 'png')

    if file and interval > 0:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with rasterio.open(filepath) as dataset:
            dem_data = dataset.read(1).astype('float32')
            transform = dataset.transform
            profile = dataset.profile
            if dataset.nodata is not None:
                dem_data[dem_data == dataset.nodata] = np.nan

        if output_format == 'png':
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contour_result.png')
            generate_colored_contour_png(dem_data, transform, output_path, interval)
            return send_file(output_path, as_attachment=True, download_name='contour_map.png')
        elif output_format == 'shp':
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contour_result.shp')
            save_contour_shapefile(dem_data, transform, dataset.crs, output_path, interval)
            # ضغط ملفات SHP في ملف zip
            import zipfile
            shapefile_base = output_path[:-4]
            zip_path = shapefile_base + '.zip'
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for ext in ['shp', 'shx', 'dbf', 'prj']:
                    filepath = shapefile_base + '.' + ext
                    if os.path.exists(filepath):
                        zipf.write(filepath, os.path.basename(filepath))
            return send_file(zip_path, as_attachment=True, download_name='contour_map.zip')
        else:
            return jsonify({"error": "Invalid output format"}), 400

    return jsonify({"error": "Invalid input"}), 400


# Slope route
@app.route("/slope", methods=['POST'])
def generate_slope():
    file = request.files['dem_file']
    output_format = request.form.get('output_format', 'png')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    with rasterio.open(filepath) as src:
        dem = src.read(1)
        transform = src.transform
        profile = src.profile

    xres, yres = transform.a, -transform.e

    # Calculate slope
    dzdx, dzdy = np.gradient(dem, xres, yres)
    slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)) * (180 / np.pi)

    if output_format == 'png':
        # Custom colormap similar to ArcGIS
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = ['#00ff00',  # 0-5
                  '#ffff00',  # 5-15
                  '#ffa500',  # 15-30
                  '#ff0000',  # 30-45
                  '#800000']  # 45-90
        bounds = [0, 5, 15, 30, 45, 90]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)

        # Draw map
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(slope, cmap=cmap, norm=norm)
        cbar = plt.colorbar(img, ticks=bounds, ax=ax)
        cbar.set_label('Slope (degrees)')
        ax.set_title('Slope Map - ArcGIS Style Colormap')
        ax.axis('off')

        # Save image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'slope_result.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        return send_file(output_path, as_attachment=True, download_name='slope_map.png')
    else:
        # Save as GeoTIFF
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'slope_result.tif')
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope.astype(rasterio.float32), 1)

        return send_file(output_path, as_attachment=True, download_name='slope_map.tif')


# Aspect route
@app.route("/aspect", methods=['POST'])
def generate_aspect_map():
    file = request.files['dem_file']
    output_format = request.form.get('output_format', 'png')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    with rasterio.open(filepath) as src:
        dem = src.read(1)
        transform = src.transform
        profile = src.profile
        pixel_size_x = src.res[0]
        pixel_size_y = src.res[1]

    # Calculate derivatives
    dy, dx = np.gradient(dem, pixel_size_y, pixel_size_x)

    # Calculate aspect
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360 + aspect_deg, aspect_deg)

    # Identify flat areas
    flat_mask = (dx == 0) & (dy == 0)
    aspect_deg[flat_mask] = -1

    # Classify directions
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

    if output_format == 'png':
        # Colors for each direction
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

        # Draw map
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

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aspect_result.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close()

        return send_file(output_path, as_attachment=True, download_name='aspect_map.png')
    else:
        # Save as GeoTIFF
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aspect_result.tif')
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(aspect_classes.astype(rasterio.uint8), 1)

        return send_file(output_path, as_attachment=True, download_name='aspect_map.tif')


# Cohere GIS chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')

    # Create a session if it doesn't exist
    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {"role": "SYSTEM",
             "message": "You are a GIS (Geographic Information System) specialist assistant. Only provide information related to GIS topics, spatial analysis, remote sensing, mapping, and geospatial technologies. If asked about non-GIS topics, politely redirect the conversation to GIS-related subjects. Be helpful, concise, and accurate in your explanations about GIS concepts, tools, and techniques. Format your responses using markdown for better readability - use headings, bullet points, code blocks, and emphasis where appropriate."}
        ]

    # Add user message to chat history
    chat_histories[session_id].append({"role": "USER", "message": user_message})

    try:
        # Get response from Cohere
        response = co.chat(
            message=user_message,
            chat_history=[{"role": m["role"], "message": m["message"]} for m in chat_histories[session_id][1:-1]],
            preamble=chat_histories[session_id][0]["message"],
            temperature=0.7,
            max_tokens=800,
        )

        # Extract the response text
        ai_message = response.text

        # Add AI response to chat history
        chat_histories[session_id].append({"role": "CHATBOT", "message": ai_message})

        # Limit history length to prevent it from growing too large
        if len(chat_histories[session_id]) > 15:
            chat_histories[session_id] = [chat_histories[session_id][0]] + chat_histories[session_id][-14:]

        return jsonify({
            "response": ai_message,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "response": "Sorry, I encountered an error processing your request. Please try again.",
            "session_id": session_id
        }), 500


if __name__ == '__main__':
    app.run(debug=True)