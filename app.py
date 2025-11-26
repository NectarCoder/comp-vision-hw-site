import base64
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

from hwsources.module1 import calculate_focal_length, calculate_real_dimension
import hwsources.module2_part1 as module2_part1
from hwsources.module2_part2 import process_image as module2_process_image
import hwsources.module2_part3 as module2_part3

# Serve static assets from the templates folder so they can be moved
# from /static into /templates while keeping the same "url_for('static', ...)"
app = Flask(__name__, static_folder='templates', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
PROJECT_ROOT = Path(__file__).resolve().parent
MODULE2_PART1_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm2'
MODULE2_PART1_SCENE = MODULE2_PART1_DIR / 'scene.jpg'
MODULE2_PART3_TEMPLATE_LIMIT = 25
INSTRUCTIONS_DIR = PROJECT_ROOT / 'hwinstructions'

MODULE_INSTRUCTION_FILES = {
    'a1': 'Assignment1.pdf',
    'module1': 'Assignment1.pdf',
    '1': 'Assignment1.pdf',
    'a2': 'Assignment2.pdf',
    'module2': 'Assignment2.pdf',
    '2': 'Assignment2.pdf',
    'a3': 'Assignment3.pdf',
    'module3': 'Assignment3.pdf',
    '3': 'Assignment3.pdf',
    'a4': 'Assignment4.pdf',
    'module4': 'Assignment4.pdf',
    '4': 'Assignment4.pdf',
    'a56': 'Assignment5-6.pdf',
    'module5': 'Assignment5-6.pdf',
    'module6': 'Assignment5-6.pdf',
    '5': 'Assignment5-6.pdf',
    '6': 'Assignment5-6.pdf',
    'a7': 'Assignment7.pdf',
    'module7': 'Assignment7.pdf',
    '7': 'Assignment7.pdf',
}


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


def _image_array_to_data_url(image, ext: str = '.png') -> str:
    """Encode an OpenCV image (np.ndarray) into a base64 data URL."""
    encode_ext = ext if ext.startswith('.') else f'.{ext}'
    mime = 'image/png'
    if encode_ext.lower() in {'.jpg', '.jpeg'}:
        mime = 'image/jpeg'

    success, buffer = cv2.imencode(encode_ext, image)
    if not success:
        raise ValueError('Failed to encode image buffer for response.')

    encoded = base64.b64encode(buffer.tobytes()).decode('ascii')
    return f"data:{mime};base64,{encoded}"


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
            matched = module2_part1.match_template(
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


def _load_module2_part3_references():
    """Return the static scene preview plus all template thumbnails."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = module2_part3.find_template_files(MODULE2_PART1_DIR, limit=MODULE2_PART3_TEMPLATE_LIMIT)
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    templates_payload = []
    for tpl_path in template_files:
        try:
            templates_payload.append({
                'filename': tpl_path.name,
                'label': tpl_path.stem,
                'image': _image_file_to_data_url(tpl_path),
            })
        except FileNotFoundError:
            continue

    return {
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': _image_file_to_data_url(MODULE2_PART1_SCENE),
        },
        'templates': templates_payload,
    }


def _run_module2_part3(threshold: float, blur_multiplier: float):
    """Execute the CLI logic from module2_part3 via server-side call."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = module2_part3.find_template_files(MODULE2_PART1_DIR, limit=MODULE2_PART3_TEMPLATE_LIMIT)
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    scene = cv2.imread(str(MODULE2_PART1_SCENE))
    if scene is None:
        raise ValueError(f"Failed to load scene image: {MODULE2_PART1_SCENE}")
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    all_detections = []
    skipped_templates = []

    for tpl_path in template_files:
        tpl = cv2.imread(str(tpl_path))
        if tpl is None:
            skipped_templates.append({'filename': tpl_path.name, 'reason': 'unreadable'})
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        boxes, scores = module2_part3.match_template_once(scene_gray, tpl_gray, threshold=threshold)
        if not boxes:
            continue

        keep_idxs = module2_part1.non_max_suppression(boxes, scores, iou_thresh=0.35)
        for idx in keep_idxs:
            all_detections.append((boxes[idx], scores[idx], tpl_path.stem))

    drawn = module2_part3.draw_detections(scene, all_detections)
    boxes_to_blur = [det[0] for det in all_detections]
    blurred = module2_part3.blur_regions(drawn, boxes_to_blur, blur_multiplier=blur_multiplier)

    detection_payload = [
        {
            'label': label,
            'score': score,
            'box': {
                'x1': box[0],
                'y1': box[1],
                'x2': box[2],
                'y2': box[3],
            }
        }
        for (box, score, label) in all_detections
    ]

    result = {
        'threshold': threshold,
        'blurMultiplier': blur_multiplier,
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': _image_file_to_data_url(MODULE2_PART1_SCENE),
        },
        'detectionsImage': _image_array_to_data_url(drawn),
        'blurredImage': _image_array_to_data_url(blurred),
        'detections': detection_payload,
        'summary': {
            'detected': len(all_detections),
            'templatesTested': len(template_files),
            'skippedTemplates': skipped_templates,
        }
    }

    if all_detections:
        result['message'] = f"Detected {len(all_detections)} region(s) with threshold {threshold:.2f}."
    else:
        result['message'] = f"No objects detected at threshold {threshold:.2f}."

    return result

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


@app.route('/instructions/<module_id>')
def module_instructions(module_id: str):
    """Serve the assignment PDF for the requested module, if it exists."""
    key = (module_id or '').lower()
    pdf_name = MODULE_INSTRUCTION_FILES.get(key)
    if not pdf_name:
        abort(404)

    pdf_path = INSTRUCTIONS_DIR / pdf_name
    if not pdf_path.exists():
        abort(404)

    return send_from_directory(INSTRUCTIONS_DIR, pdf_name)


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


@app.route('/api/a2/part3/references', methods=['GET'])
def get_a2_part3_references():
    try:
        refs = _load_module2_part3_references()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load references: {exc}'}), 500

    return jsonify(refs)


@app.route('/api/a2/part3', methods=['POST'])
def handle_a2_part3_run():
    data = request.get_json(silent=True) or {}

    threshold = data.get('threshold', 0.75)
    blur_multiplier = data.get('blurMultiplier', 2.5)

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({'error': 'Threshold must be a numeric value between 0.0 and 1.0.'}), 400

    try:
        blur_multiplier = float(blur_multiplier)
    except (TypeError, ValueError):
        return jsonify({'error': 'Blur multiplier must be a positive numeric value.'}), 400

    if not 0.0 <= threshold <= 1.0:
        return jsonify({'error': 'Threshold must be between 0.0 and 1.0.'}), 400
    if blur_multiplier <= 0 or blur_multiplier > 25:
        return jsonify({'error': 'Blur multiplier must be between 0 and 25.'}), 400

    try:
        results = _run_module2_part3(threshold, blur_multiplier)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 2 Part 3: {exc}'}), 500

    return jsonify(results)

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
        # If the query contains a truthy `download` flag, serve the file as an attachment
        download_flag = request.args.get('download', '').lower()
        as_attachment = download_flag in ('1', 'true', 'yes', 'on')
        # Flask's send_from_directory supports `as_attachment` which will set
        # the Content-Disposition header and prompt a file download in the
        # user's browser. Still, the UI also uses an HTML anchor with the
        # download attribute for a direct client-side fallback.
        return send_from_directory('hwsources', filename, mimetype='text/plain', as_attachment=as_attachment)
    except Exception:
        abort(404)

if __name__ == '__main__':
    # Running on 0.0.0.0 ensures it's accessible externally on your Azure VM
    app.run(host='0.0.0.0', port=5000, debug=True)