import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from movenet_pre_processor.normalizer import normalize_all_frames
from movenet_pre_processor.pre_processor import build_sequence
from movenet_pre_processor.custom_layers import SqueezeLayer

app = Flask(__name__)

# Global variables to store the model and maps
model = None
label_map = None
angle_map = None
idx_to_label = None
idx_to_angle = None


# Load model and maps once at startup
def load_resources(model_path, label_map_path, angle_map_path):
    global model, label_map, angle_map, idx_to_label, idx_to_angle

    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects={"SqueezeLayer": SqueezeLayer})

    # Load maps
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    with open(angle_map_path, "r") as f:
        angle_map = json.load(f)

    # Invert label map for decoding
    idx_to_label = {v: k for k, v in label_map.items()}
    idx_to_angle = {v: k for k, v in angle_map.items()}


# Prediction function that uses the global model
def predict_throw(json_path, angle_label):
    global model, label_map, angle_map, idx_to_label

    # Load and preprocess clip
    with open(json_path, "r") as f:
        raw_frames = json.load(f)

    normalized = normalize_all_frames(raw_frames)
    sequence = build_sequence(
        frames=normalized,
        max_people=2,
        target_len=60,
        fill_mode="last"
    )

    # Prepare model inputs
    X = np.expand_dims(sequence, axis=0)  # shape: (1, 60, 68)
    angle_index = np.array([[angle_map[angle_label]]])  # shape: (1, 1)

    # Run prediction
    probs = model.predict({"pose_input": X, "angle_input": angle_index})
    predicted_class = int(np.argmax(probs))

    return {
        "predicted_label": idx_to_label[predicted_class],
        "confidence": float(np.max(probs)),
        "all_probs": probs.tolist()[0],
        "class_index": predicted_class
    }


# Initialize the app with the model and maps
def initialize():
    model_path = "models/throw_detection_v0-1.keras"
    label_map_path = "maps/label_map.json"
    angle_map_path = "maps/angle_map.json"
    load_resources(model_path, label_map_path, angle_map_path)


with app.app_context():
    initialize()


# API endpoint
@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    raw_frames = data['frames']  # Pass frames directly instead of file path
    # angle_label = data['angle_label']
    angle_label = 'side'  # For now, I am doing this until I have a solid angle detection model.

    # Modified prediction function
    result = predict_throw_direct(raw_frames, angle_label)
    return jsonify(result)


def predict_throw_direct(raw_frames, angle_label):
    global model, label_map, angle_map, idx_to_label

    normalized = normalize_all_frames(raw_frames)
    sequence = build_sequence(
        frames=normalized,
        max_people=2,
        target_len=60,
        fill_mode="last"
    )

    # Rest of the prediction code remains the same
    X = np.expand_dims(sequence, axis=0)
    angle_index = np.array([[angle_map[angle_label]]])
    probs = model.predict({"pose_input": X, "angle_input": angle_index})
    predicted_class = int(np.argmax(probs))

    return {
        "predicted_label": idx_to_label[predicted_class],
        "confidence": float(np.max(probs)),
        "all_probs": probs.tolist()[0],
        "class_index": predicted_class
    }


if __name__ == '__main__':
    app.run(debug=False)
