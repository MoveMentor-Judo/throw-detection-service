import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from normalizer import normalize_all_frames
from pre_processor import build_sequence


def build_label_maps(metadata: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    labels = sorted(set(entry["label"] for entry in metadata))
    angles = sorted(set(entry["angle"] for entry in metadata))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    angle_to_idx = {angle: i for i, angle in enumerate(angles)}
    return label_to_idx, angle_to_idx


def build_dataset(
        metadata: List[Dict],
        clip_dir: Path,
        label_to_idx: Dict[str, int],
        angle_to_idx: Dict[str, int],
        max_people: int = 2,
        target_len: int = 60,
        fill_mode: str = "last"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, a = [], [], []

    for entry in metadata:
        clip_path = clip_dir / entry["file"]
        if not clip_path.exists():
            print(f"File not found: {clip_path}")
            continue

        try:
            with open(clip_path, "r") as f:
                raw_frames = json.load(f)

            normalized = normalize_all_frames(raw_frames)
            sequence = build_sequence(
                frames=normalized,
                max_people=max_people,
                target_len=target_len,
                fill_mode=fill_mode
            )

            X.append(sequence)
            y.append(label_to_idx[entry["label"]])
            a.append(angle_to_idx[entry["angle"]])

        except Exception as e:
            print(f"Failed to process {clip_path.name}: {e}")

    return np.stack(X), np.array(y), np.array(a)


def save_dataset(X: np.ndarray, y: np.ndarray, angles: np.ndarray, output_path: Path):
    np.savez(output_path, X=X, y=y, angles=angles)
    print(f"Saved dataset to {output_path} — X: {X.shape}, y: {y.shape}, angles: {angles.shape}")


def save_label_maps(label_map: Dict[str, int], angle_map: Dict[str, int], out_dir: Path):
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(out_dir / "angle_map.json", "w") as f:
        json.dump(angle_map, f, indent=2)
    print("Saved label_map.json and angle_map.json")



