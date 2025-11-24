import base64
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

from hwsources.module1 import calculate_focal_length, calculate_real_dimension
from hwsources.module2_part1 import match_template
from hwsources.module2_part2 import process_image as module2_process_image

# Serve static assets from the templates folder so they can be moved
# from /static into /templates while keeping the same "url_for('static', ...)"
app = Flask(__name__, static_folder='templates', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
PROJECT_ROOT = Path(__file__).resolve().parent
MODULE2_PART1_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm2'
MODULE2_PART1_SCENE = MODULE2_PART1_DIR / 'scene.jpg'


def _image_file_to_data_url(path: Path) -> str:
    """Return a base64 data URL for the given image path."""
    suffix = path.suffix.lower()
    mime = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff'
    }.get(suffix, 'application/octet-stream')

    data = path.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{encoded}"


def _clear_module2_part1_results() -> int:
    """Remove generated match images from the Module 2 Part 1 resources folder."""
    if not MODULE2_PART1_DIR.exists():
        return 0

    deleted = 0
    for path in MODULE2_PART1_DIR.glob('*_matched_template_*'):
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _run_module2_part1(threshold: float):
    """Execute template matching for every template_* file and collect results."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = sorted(MODULE2_PART1_DIR.glob('template_*.png'))
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    _clear_module2_part1_results()

    matches = []
    for tpl in template_files:
        tpl_name = tpl.name
        output_path = MODULE2_PART1_SCENE.with_name(
            f"{MODULE2_PART1_SCENE.stem}_matched_{tpl.stem}{MODULE2_PART1_SCENE.suffix}"
        )

        try:
            matched = match_template(
                scene_path=str(MODULE2_PART1_SCENE),
                template_path=str(tpl),
                threshold=threshold,
                save_result=True,
                draw_all=True,
            )
        except Exception as exc:
            matches.append({
                'template': tpl_name,
                'matched': False,
                'error': str(exc),
                'outputImage': None,
            })
            continue

        if matched and output_path.exists():
            image_data = _image_file_to_data_url(output_path)
        else:
            image_data = None

        matches.append({
            'template': tpl_name,
            'matched': bool(matched and image_data),
            'error': None,
            'outputImage': image_data,
            'outputFilename': output_path.name if image_data else None,
        })

    scene_data = _image_file_to_data_url(MODULE2_PART1_SCENE)

    matched_count = sum(1 for m in matches if m['matched'])
    return {
        'threshold': threshold,
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': scene_data,
        },
        'matches': matches,
        'summary': {
            'matched': matched_count,
            'total': len(matches),
        }
    }

# --- Routes ---

@app.route('/')
def home():
    """Serves the main frontend HTML page."""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve the favicon from the static folder at the root path /favicon.ico.

    Browsers usually request /favicon.ico — Flask serves static files under /static/, so
    we add this small route so the favicon is available at the expected path.
    """
    return send_from_directory(app.static_folder, 'favicon.ico')


def _parse_numeric(data, key, allow_zero=False):
    """Try to coerce JSON field `key` into float, raising ValueError on failure."""
    if key not in data:
        raise ValueError(f"Missing field: {key}")
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be numeric")

    if not allow_zero and value == 0:
        raise ValueError(f"{key} must be non-zero")
    if value < 0:
        raise ValueError(f"{key} must be positive")
    return value


@app.route('/api/a1/focal-length', methods=['POST'])
def module1_calculate_focal_length():
    data = request.get_json(silent=True) or {}
    try:
        pixel_width = _parse_numeric(data, 'pixelWidth')
        real_width = _parse_numeric(data, 'realWidth')
        distance = _parse_numeric(data, 'distance')
        focal_length = calculate_focal_length(pixel_width, real_width, distance)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'focalLength': focal_length,
        'refPixelWidth': pixel_width,
        'refRealWidth': real_width,
        'refDistance': distance
    })


@app.route('/api/a1/real-width', methods=['POST'])
def module1_calculate_real_width():
    data = request.get_json(silent=True) or {}
    try:
        pixel_width = _parse_numeric(data, 'pixelWidth')
        focal_length = _parse_numeric(data, 'focalLength')
        distance = _parse_numeric(data, 'distance')
        real_width = calculate_real_dimension(pixel_width, focal_length, distance)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'realWidth': real_width,
        'testPixelWidth': pixel_width,
        'testDistance': distance,
        'focalLength': focal_length
    })

@app.route('/api/a2', methods=['POST'])
def handle_a2():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 2 code
    response = f"[Stub] Module 2 executed on input length: {len(user_input)}"
    return jsonify({'result': response})


@app.route('/api/a2/part1', methods=['POST'])
def handle_a2_part1_run():
    data = request.get_json(silent=True) or {}
    threshold = data.get('threshold', 0.8)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({'error': 'Threshold must be a numeric value between 0.0 and 1.0.'}), 400

    if not 0.0 <= threshold <= 1.0:
        return jsonify({'error': 'Threshold must be between 0.0 and 1.0.'}), 400

    try:
        results = _run_module2_part1(threshold)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Failed to run template matching: {exc}'}), 500

    return jsonify(results)


@app.route('/api/a2/part1/scene', methods=['GET'])
def get_a2_part1_scene():
    if not MODULE2_PART1_SCENE.exists():
        return jsonify({'error': f"Scene file missing at {MODULE2_PART1_SCENE}"}), 404

    return jsonify({
        'filename': MODULE2_PART1_SCENE.name,
        'image': _image_file_to_data_url(MODULE2_PART1_SCENE),
    })


@app.route('/api/a2/part1/results', methods=['DELETE', 'POST'])
def clear_a2_part1_results():
    deleted = _clear_module2_part1_results()
    return jsonify({'deleted': deleted})


@app.route('/api/a2/part2', methods=['POST'])
def handle_a2_part2_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename supplied'}), 400

    filename = secure_filename(image_file.filename) or 'upload.png'
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        with TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            input_path = tmp_dir_path / filename
            image_file.save(input_path)

            blur_override = tmp_dir_path / f"{input_path.stem}_blurred.png"
            restored_override = tmp_dir_path / f"{input_path.stem}_restored.png"

            (
                blur_path,
                restored_path,
                kernel_used,
                sigma_used,
            ) = module2_process_image(
                input_image=input_path,
                kernel_size=None,
                sigma=None,
                balance=1e-2,
                eps=1e-6,
                blur_output=str(blur_override),
                restored_output=str(restored_override),
            )

            def _path_to_data_url(image_path: Path) -> str:
                data = image_path.read_bytes()
                encoded = base64.b64encode(data).decode('ascii')
                return f"data:image/png;base64,{encoded}"

            return jsonify({
                'blurredImage': _path_to_data_url(Path(blur_path)),
                'restoredImage': _path_to_data_url(Path(restored_path)),
                'originalFilename': filename,
                'kernelSize': kernel_used,
                'sigma': sigma_used,
            })
    except FileNotFoundError:
        return jsonify({'error': 'Uploaded image could not be processed'}), 400
    except Exception as exc:
        return jsonify({'error': f'Processing failed: {exc}'}), 500

@app.route('/api/a3', methods=['POST'])
def handle_a3():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 3 code
    response = f"[Stub] Module 3 received data."
    return jsonify({'result': response})

@app.route('/api/a4', methods=['POST'])
def handle_a4():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 4 code
    response = f"[Stub] Module 4 logic placeholder."
    return jsonify({'result': response})

@app.route('/api/a56', methods=['POST'])
def handle_a56():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 5 & 6 code
    response = f"[Stub] Module 5 & 6 complex processing placeholder."
    return jsonify({'result': response})

@app.route('/api/a7', methods=['POST'])
def handle_a7():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 7 code
    response = f"[Stub] Module 7 final project placeholder."
    return jsonify({'result': response})


@app.route('/source/<path:filename>')
def serve_source(filename):
    """Serve raw source files from the hwsources/ directory.

    This implements a small, guarded endpoint so client-side UI can fetch
    Python source (or other text files) dynamically. It strictly prevents
    path traversal and only serves files with a safe extension.
    """
    # guard against directory traversal
    if filename.startswith('/') or '..' in filename:
        abort(404)

    # restrict to safe file types
    allowed_ext = {'.py', '.txt', '.md'}
    ext = None
    try:
        import os
        _, ext = os.path.splitext(filename)
    except Exception:
        abort(404)

    if ext not in allowed_ext:
        abort(404)

    # deliver the file contents from the hwsources directory
    try:
        return send_from_directory('hwsources', filename, mimetype='text/plain')
    except Exception:
        abort(404)

if __name__ == '__main__':
    # Running on 0.0.0.0 ensures it's accessible externally on your Azure VM
    app.run(host='0.0.0.0', port=5000, debug=True)