"""LIBERO environment helpers, adapted from OpenVLA's experiments/robot/libero/libero_utils.py without the TensorFlow
dependency (JPEG round-trip + Lanczos resize done with PIL). Same 180° rotation of the agentview camera."""
from __future__ import annotations

import io, math, os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
NUM_STEPS_WAIT = 10  # let objects settle after reset (OpenVLA convention)
MAX_STEPS = {"libero_spatial": 500, "libero_object": 500, "libero_goal": 500, "libero_10": 500, "libero_90": 500}  # NORA eval uses 500 for every suite


def load_suite(name: str):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[name]()


def make_env(task, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # OpenVLA: seed affects object positions even with a fixed initial state
    return env


def raw_frame(obs) -> np.ndarray:
    """agentview camera, rotated 180° to match training preprocessing. uint8 (H, W, 3)."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def policy_image(obs, size: int = 224) -> np.ndarray:
    """OpenVLA/NORA preprocessing: rotate, JPEG encode/decode, Lanczos resize to size×size."""
    img = Image.fromarray(raw_frame(obs))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=95); buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper(action: np.ndarray) -> np.ndarray:
    """NORA's LIBERO eval (run_libero_eval_nora.py, model_family='openvla'): gripper [0,1] → sign(2x−1) ∈ {−1,+1},
    then INVERT (RLDS aligns 0=close/1=open; the env wants −1=open/+1=close), then >=0 → +1. The inversion was
    missing in our first base-policy run and halved the baseline success rate."""
    action = np.array(action, dtype=np.float64)
    action[..., -1] = np.sign(2.0 * action[..., -1] - 1.0)
    action[..., -1] = action[..., -1] * -1.0
    action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
    return action


def save_video(frames: list[np.ndarray], path: str | Path, fps: int = 30) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()
    return path
