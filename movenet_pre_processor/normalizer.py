import numpy as np
from typing import List, Dict

def normalize_pose(pose: dict) -> dict:
    """
    Normalize a single pose object and return a new pose dictionary.
    Keypoints will be translated and scaled based on torso/hip reference.

    Parameters:
        pose (dict): A dictionary containing 'keypoints', 'id', 'score', etc.

    Returns:
        dict: A new pose dict with normalized 'x' and 'y' in 'keypoints'
    """
    keypoints = pose["keypoints"]
    kp_dict = {kp["name"]: (kp["x"], kp["y"]) for kp in keypoints}

    # Define reference joints
    reference_joints = ("left_hip", "right_hip")
    scale_joints = ("left_shoulder", "right_hip")

    # Compute center reference (e.g., mid-hip)
    if all(j in kp_dict for j in reference_joints):
        ref = np.mean([kp_dict[j] for j in reference_joints], axis=0)
    else:
        ref = np.array([0.0, 0.0])  # fallback

    # Translate keypoints
    translated = []
    for kp in keypoints:
        x, y = kp["x"] - ref[0], kp["y"] - ref[1]
        translated.append((x, y))

    # Compute scale based on body length
    if all(j in kp_dict for j in scale_joints):
        p1 = np.array(kp_dict[scale_joints[0]])
        p2 = np.array(kp_dict[scale_joints[1]])
        scale = np.linalg.norm(p1 - p2)
    else:
        scale = 1.0

    if scale == 0:
        scale = 1.0

    # Build new keypoints
    normalized_kps = []
    for i, kp in enumerate(keypoints):
        norm_x = translated[i][0] / scale
        norm_y = translated[i][1] / scale
        normalized_kps.append({
            "x": float(norm_x),
            "y": float(norm_y),
            "score": kp["score"],
            "name": kp["name"]
        })

    # Return new normalized pose object
    return {
        "id": pose.get("id", None),
        "keypoints": normalized_kps,
        "score": pose.get("score", None),
        "box": pose.get("box", None)
    }


def normalize_all_frames(frames: List[dict]) -> List[dict]:
    """
    Normalize all poses in all frames.

    Parameters:
        frames (list): List of frames with structure:
            [{ "timestamp": ..., "poses": [ { "keypoints": [...] }, ... ] }, ...]

    Returns:
        list: Same structure with normalized keypoints
    """
    normalized_frames = []
    for frame in frames:
        norm_poses = [normalize_pose(pose) for pose in frame["poses"]]
        normalized_frames.append({
            "timestamp": frame["timestamp"],
            "poses": norm_poses
        })
    return normalized_frames

