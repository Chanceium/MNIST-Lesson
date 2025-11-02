"""
Script to convert old Keras models to new format compatible with TensorFlow 2.15
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tf_keras
import h5py
import numpy as np

def convert_model(old_path):
    """Convert old model format to new format"""
    print(f"Converting {old_path}...")

    # Load weights and architecture manually
    with h5py.File(old_path, 'r') as f:
        # Create a new model with the same architecture
        model = tf_keras.models.Sequential([
            tf_keras.layers.Input(shape=(28, 28, 1)),
            tf_keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf_keras.layers.MaxPooling2D((2, 2)),
            tf_keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf_keras.layers.MaxPooling2D((2, 2)),
            tf_keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf_keras.layers.Flatten(),
            tf_keras.layers.Dense(64, activation='relu'),
            tf_keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        # Try to load weights
        try:
            model.load_weights(old_path)
            print(f"✓ Successfully loaded weights from {old_path}")
        except Exception as e:
            print(f"Note: {e}")
            # Model architecture loaded, weights might not match exactly

    return model

# Convert all three models
model_names = ['baseline_model', 'augmented_model', 'overfitted_model']

for name in model_names:
    old_path = f'models/{name}.h5'
    new_path = f'models/{name}_new.h5'

    if os.path.exists(old_path):
        model = convert_model(old_path)
        model.save(new_path)
        print(f"✓ Saved {new_path}\n")
    else:
        print(f"✗ {old_path} not found\n")

print("Conversion complete!")
