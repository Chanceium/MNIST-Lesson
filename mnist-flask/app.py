from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import io
import base64
import cv2

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

def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Grad-CAM for (28,28,1) inputs. Returns a uint8 heatmap (28x28).
    Works with Keras 3 even if the loaded model has never been called.
    """
    # 1) Find last conv layer
    last_conv_layer = None
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.__class__.__name__.lower():
                last_conv_layer = layer
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        return np.zeros((28, 28), dtype=np.uint8)

    if last_conv_layer is None:
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except:
            return np.zeros((28, 28), dtype=np.uint8)

    # 2) Rebuild a functional graph from a fresh Input, capturing conv features and logits
    inp = keras.Input(shape=(28, 28, 1))
    x = inp
    conv_activations = None
    for layer in model.layers:
        x = layer(x)  # reuse the same layer objects & weights
        if layer.name == last_conv_layer_name:
            conv_activations = x

    if conv_activations is None:
        return np.zeros((28, 28), dtype=np.uint8)

    feature_extractor = keras.Model(inp, [conv_activations, x])  # [features, logits]

    # 3) Forward/Backward pass
    with tf.GradientTape() as tape:
        features, preds = feature_extractor(img_array, training=False)
        top_index = tf.argmax(preds[0])
        top_score = preds[:, top_index]
        tape.watch(features)

    grads = tape.gradient(top_score, features)
    if grads is None:
        return np.zeros((28, 28), dtype=np.uint8)

    # 4) Importance weights & heatmap
    # pooled_grads: (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    features = features[0].numpy()          # (H, W, C)
    weights = pooled_grads.numpy()          # (C,)

    for c in range(features.shape[-1]):
        features[:, :, c] *= weights[c]

    heatmap = np.mean(features, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    # Normalize and resize to 28x28
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (28, 28)).astype(np.float32)

    return np.uint8(255 * heatmap)


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

            # Generate Grad-CAM heatmap
            try:
                heatmap = generate_gradcam(model, processed_image)
                # Convert heatmap to base64 for transmission
                heatmap_img = Image.fromarray(heatmap)
                buffer = io.BytesIO()
                heatmap_img.save(buffer, format='PNG')
                heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                heatmap_data = f'data:image/png;base64,{heatmap_base64}'
            except Exception as e:
                print(f"Error generating Grad-CAM for {model_name}: {e}")
                heatmap_data = None

            predictions[model_name] = {
                'digit': predicted_digit,
                'confidence': confidence,
                'probabilities': all_probs,
                'gradcam': heatmap_data
            }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
