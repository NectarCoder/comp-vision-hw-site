from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from hwsources.module1 import calculate_focal_length, calculate_real_dimension

# Serve static assets from the templates folder so they can be moved
# from /static into /templates while keeping the same "url_for('static', ...)"
app = Flask(__name__, static_folder='templates', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

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

if __name__ == '__main__':
    # Running on 0.0.0.0 ensures it's accessible externally on your Azure VM
    app.run(host='0.0.0.0', port=5000, debug=True)