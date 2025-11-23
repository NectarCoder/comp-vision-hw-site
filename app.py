from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import time

app = Flask(__name__)
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

# --- Assignment Stubs ---
# These are the endpoints your Javascript will call.
# You will eventually import your actual homework scripts here.

@app.route('/api/a1', methods=['POST'])
def handle_a1():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect your actual Assignment 1 Python code here
    # Example: result = assignment1.run(user_input)
    
    # Simulating processing time
    time.sleep(0.5) 
    
    response = f"[Stub] Assignment 1 processed: '{user_input}'\n(Replace this logic in app.py with your real CV code)"
    return jsonify({'result': response})

@app.route('/api/a2', methods=['POST'])
def handle_a2():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Assignment 2 code
    response = f"[Stub] Assignment 2 executed on input length: {len(user_input)}"
    return jsonify({'result': response})

@app.route('/api/a3', methods=['POST'])
def handle_a3():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Assignment 3 code
    response = f"[Stub] Assignment 3 received data."
    return jsonify({'result': response})

@app.route('/api/a4', methods=['POST'])
def handle_a4():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Assignment 4 code
    response = f"[Stub] Assignment 4 logic placeholder."
    return jsonify({'result': response})

@app.route('/api/a56', methods=['POST'])
def handle_a56():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Assignment 5 & 6 code
    response = f"[Stub] Assignment 5 & 6 complex processing placeholder."
    return jsonify({'result': response})

@app.route('/api/a7', methods=['POST'])
def handle_a7():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Assignment 7 code
    response = f"[Stub] Assignment 7 final project placeholder."
    return jsonify({'result': response})

if __name__ == '__main__':
    # Running on 0.0.0.0 ensures it's accessible externally on your Azure VM
    app.run(host='0.0.0.0', port=5000, debug=True)