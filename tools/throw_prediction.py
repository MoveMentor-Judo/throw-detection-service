import json
import numpy as np
import tensorflow as tf
from movenet_pre_processor.normalizer import normalize_all_frames
from movenet_pre_processor.pre_processor import build_sequence
from movenet_pre_processor.custom_layers import SqueezeLayer


# === Load maps once ===
def load_maps(label_map_path, angle_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    with open(angle_map_path, "r") as f:
        angle_map = json.load(f)
    # Invert label map for decoding
    idx_to_label = {v: k for k, v in label_map.items()}
    idx_to_angle = {v: k for k, v in angle_map.items()}
    return label_map, angle_map, idx_to_label, idx_to_angle

# === Inference function ===
def predict_throw(
        model_path: str,
        json_path: str,
        angle_label: str,
        label_map_path: str = "maps/label_map.json",
        angle_map_path: str = "maps/angle_map.json"
):
    # Load model and maps
    model = tf.keras.models.load_model(model_path, custom_objects={"SqueezeLayer": SqueezeLayer})
    label_map, angle_map, idx_to_label, _ = load_maps(label_map_path, angle_map_path)

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
