from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load your trained fish identification model
model = load_model('fishcom_model.h5')  # Ensure correct path to your model

# A dictionary mapping class indices to fish species names (adjusted to match your class names)
FISH_LABELS = {
    0: 'Angelfish',
    1: 'Betaa raja',
    2: 'Betta Bellica',
    3: 'Betta Brownorum',
    4: 'Betta Ocellata',
    5: 'Betta coccina',
    6: 'Betta enisae',
    7: 'Betta imbellis',
    8: 'Betta mahachaiensis',
    9: 'Betta persephone',
    10: 'Betta picta',
    11: 'Betta smaragdina',
    12: 'Betta spilotgena',
    13: 'Betta splendens',
    14: 'Bluegill Sunfish',
    15: 'Cherry Barb',
    16: 'Clarias batrachus',
    17: 'Clown loach',
    18: 'Glosssogobious aurues',
    19: 'Guppy',
    20: 'Molly',
    21: 'Neon Tetra',
    22: 'Panda Corydoras',
    23: 'Sinarapan',
    24: 'Swordtail',
    25: 'Zebra Danio',
    26: 'Zebra pleco',
    27: 'Goldfish'
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a valid image.'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((256, 256))  # Resize as per your training configuration
        img_array = np.array(img) / 255.0  # Ensure this matches your training preprocessing
        
        # Add check to see if image is in the right shape
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension if missing
        else:
            raise ValueError("Image array does not have 3 dimensions.")

        predictions = model.predict(img_array)
        print("Raw predictions:", predictions)  # This should show the raw scores
        predicted_class_idx = np.argmax(predictions, axis=-1)[0]

        # Print out the predicted index and its corresponding species
        predicted_species = FISH_LABELS.get(predicted_class_idx, "Unknown Species")
        print(f"Predicted Class Index: {predicted_class_idx}, Predicted Species: {predicted_species}")

        return jsonify({'predicted_class_index': int(predicted_class_idx), 'predicted_species': predicted_species})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image.'}), 500



def allowed_file(filename):
    """Helper function to check if the uploaded file is a valid image type."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5050)
