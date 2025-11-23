from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import time

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

# --- Module Stubs ---
# These are the endpoints your Javascript will call.
# You will eventually import your actual homework scripts here.

@app.route('/api/a1', methods=['POST'])
def handle_a1():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect your actual Module 1 Python code here
    # Example: result = module1.run(user_input)
    
    # Simulating processing time
    time.sleep(0.5) 
    
    response = f"[Stub] Module 1 processed: '{user_input}'\n(Replace this logic in app.py with your real CV code)"
    return jsonify({'result': response})

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