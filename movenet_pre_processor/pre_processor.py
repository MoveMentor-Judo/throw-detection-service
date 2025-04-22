from typing import List, Dict, Optional
import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def flatten_pose(pose: Dict, fill_from: Optional[Dict] = None) -> List[float]:
    """
    Flattens a pose dict into a list [x1, y1, ..., x17, y17].
    If keypoints are missing, tries to fill them from `fill_from` if provided.

    Parameters:
        pose (dict): Normalized pose with "keypoints" field
        fill_from (dict): Optional pose to pull missing keypoints from

    Returns:
        List[float]: Flattened keypoint coordinates
    """
    kp_map = {kp["name"]: (kp["x"], kp["y"]) for kp in pose.get("keypoints", [])}
    fallback_map = {kp["name"]: (kp["x"], kp["y"]) for kp in fill_from.get("keypoints", [])} if fill_from else {}

    flattened = []
    for name in KEYPOINT_NAMES:
        if name in kp_map:
            x, y = kp_map[name]
        elif name in fallback_map:
            x, y = fallback_map[name]
        else:
            x, y = 0.0, 0.0
        flattened.extend([x, y])

    return flattened


def pad_or_truncate(seq: List[List[float]], target_len: int = 60) -> np.ndarray:
    """
    Pads or truncates a sequence of pose vectors to fixed length.

    Parameters:
        seq (List[List[float]]): List of frame vectors (flattened keypoints)
        target_len (int): Desired final length

    Returns:
        np.ndarray: Array of shape [target_len, vector_length]
    """
    current_len = len(seq)
    feature_dim = len(seq[0]) if current_len > 0 else 0

    if current_len >= target_len:
        return np.array(seq[:target_len])

    padded_seq = np.zeros((target_len, feature_dim), dtype=np.float32)
    padded_seq[:current_len] = seq
    return padded_seq


def build_sequence(
    frames: List[dict],
    max_people: int = 2,
    target_len: int = 60,
    fill_mode: str = "last"
) -> np.ndarray:
    """
    Constructs a [target_len, max_people * 34] sequence from frame data.

    Parameters:
        frames (list): List of frames, each with "poses" (normalized keypoints)
        max_people (int): Max people to include per frame
        target_len (int): Output sequence length
        fill_mode (str): "zero" (default) or "last" — fills missing poses/keypoints

    Returns:
        np.ndarray: Preprocessed sequence for model input
    """
    sequence = []
    last_known_poses = [None] * max_people

    for frame in frames:
        poses = frame.get("poses", [])[:max_people]
        frame_vec = []

        for i in range(max_people):
            if i < len(poses):  # Pose exists
                pose = poses[i]
                fill_pose = last_known_poses[i] if fill_mode == "last" else None
                flat = flatten_pose(pose, fill_from=fill_pose)
                last_known_poses[i] = pose
            else:  # Pose missing
                if fill_mode == "last" and last_known_poses[i]:
                    flat = flatten_pose(last_known_poses[i], fill_from=last_known_poses[i])
                else:
                    flat = [0.0] * 34
            frame_vec.extend(flat)

        sequence.append(frame_vec)

    return pad_or_truncate(sequence, target_len=target_len)

