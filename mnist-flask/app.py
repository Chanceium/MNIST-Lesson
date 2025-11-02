from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import io
import base64

app = Flask(__name__)

# Force CPU usage for TensorFlow
tf.config.set_visible_devices([], 'GPU')

# Load the three models (Keras 3.x format)
try:
    models = {
        'baseline': keras.saving.load_model('models/baseline_model.h5'),
        'augmented': keras.saving.load_model('models/augmented_model.h5'),
        'overfitted': keras.saving.load_model('models/overfitted_model.h5')
    }
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = {}

def preprocess_image(image_data):
    """Preprocess the canvas image for prediction"""
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Canvas is already 28x28, no need to resize
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0

    # Image is already white-on-black (MNIST format), no need to invert

    # Reshape for model input (batch_size, height, width, channels)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lesson')
def lesson():
    return render_template('lesson.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Get predictions from all three models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(processed_image, verbose=0)
            predicted_digit = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]) * 100)

            # Get all probabilities
            all_probs = [float(p * 100) for p in prediction[0]]

            predictions[model_name] = {
                'digit': predicted_digit,
                'confidence': confidence,
                'probabilities': all_probs
            }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
