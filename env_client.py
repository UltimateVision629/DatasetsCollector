"""
Unity LIBERO environment client via TCP (TrainingServer port 5556).

Provides a Gymnasium-compatible API for RL/BC training.
Matches VLA-Adapter's LIBERO observation/action format:
  - agentview_image: (224, 224, 3) uint8
  - robot0_joint_pos: [7] float32
  - robot0_eef_pos: [3] float32
  - robot0_eef_quat: [4] float32
  - robot0_gripper_qpos: [2] float32
  - action: [14] float32  (dual-arm EEF delta: 7 per arm [dx,dy,dz,dRx,dRy,dRz,gripper])
"""

import base64
import io
import json
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


class LiberoUnityEnv:
    """Gymnasium-style wrapper around Unity LIBERO environment via TCP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        image_size: int = 224,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.image_size = image_size
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    # ── Connection management ──────────────────────────────────────────

    def connect(self):
        """Connect to Unity TrainingServer. Blocks until Unity is ready."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self.timeout)
        print(f"[EnvClient] Connecting to {self.host}:{self.port} ...")
        while True:
            try:
                self._sock.connect((self.host, self.port))
                break
            except (ConnectionRefusedError, OSError):
                print("  Waiting for Unity TrainingServer ...")
                time.sleep(1.0)
        print("[EnvClient] Connected.")

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, msg: dict) -> dict:
        """Send JSON command, receive JSON response."""
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self._sock.sendall(data)

        # Read until newline
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Unity disconnected")
            self._buf += chunk

        line, self._buf = self._buf.split(b"\n", 1)
        raw = line.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"[EnvClient] JSON parse error. Raw response (first 500 chars): {raw[:500]}")
            raise

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment, return first observation."""
        resp = self._send({"cmd": "reset"})
        if "error" in resp:
            raise RuntimeError(f"Unity reset failed: {resp['error']}")
        return self._decode_obs(resp)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""
        action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)
        resp = self._send({"cmd": "step", "action": action_list})
        if "error" in resp:
            raise RuntimeError(f"Unity step failed: {resp['error']}")

        obs = self._decode_obs(resp["obs"])
        reward = float(resp.get("reward", 0.0))
        done = bool(resp.get("done", False))
        info = {
            "step": resp.get("step", 0),
            "success": resp.get("success", False),
        }
        return obs, reward, done, info

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation without stepping."""
        resp = self._send({"cmd": "get_obs"})
        if "error" in resp:
            raise RuntimeError(f"Unity get_obs failed: {resp['error']}")
        return self._decode_obs(resp)

    def get_task(self) -> str:
        """Get language instruction for the current task."""
        resp = self._send({"cmd": "get_task"})
        return resp.get("language_instruction", "")

    # ── Observation decoding ───────────────────────────────────────────

    def _decode_obs(self, raw: dict) -> Dict[str, np.ndarray]:
        """Decode JSON observation → numpy dict matching VLA-Adapter format."""
        obs = {}

        # Agentview image: base64 PNG → numpy
        if "agentview_image" in raw and raw["agentview_image"]:
            img_bytes = base64.b64decode(raw["agentview_image"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img_array = np.array(img, dtype=np.uint8)
            # 180° rotation to match VLA-Adapter preprocessing (libero_utils.py:35)
            img_array = img_array[::-1, ::-1].copy()
            obs["agentview_image"] = img_array

        # Eye-in-hand image (optional)
        if "eye_in_hand_image" in raw and raw["eye_in_hand_image"]:
            img_bytes = base64.b64decode(raw["eye_in_hand_image"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img_array = np.array(img, dtype=np.uint8)
            img_array = img_array[::-1, ::-1].copy()
            obs["eye_in_hand_image"] = img_array

        # Proprioception
        obs["robot0_joint_pos"] = np.array(raw.get("robot0_joint_pos", [0.0] * 6), dtype=np.float32)
        obs["robot0_eef_pos"] = np.array(raw.get("robot0_eef_pos", [0.0] * 3), dtype=np.float32)
        obs["robot0_eef_quat"] = np.array(raw.get("robot0_eef_quat", [0.0] * 4), dtype=np.float32)
        obs["robot0_gripper_qpos"] = np.array(raw.get("robot0_gripper_qpos", [0.0] * 2), dtype=np.float32)

        obs["robot1_joint_pos"] = np.array(raw.get("robot1_joint_pos", [0.0] * 6), dtype=np.float32)
        obs["robot1_eef_pos"] = np.array(raw.get("robot1_eef_pos", [0.0] * 3), dtype=np.float32)
        obs["robot1_eef_quat"] = np.array(raw.get("robot1_eef_quat", [0.0] * 4), dtype=np.float32)
        obs["robot1_gripper_qpos"] = np.array(raw.get("robot1_gripper_qpos", [0.0] * 2), dtype=np.float32)

        return obs

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


# ── Action normalization (matching VLA-Adapter's BOUNDS_Q99) ──────────

def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize action to [-1, 1] using q01/q99 bounds."""
    low = np.array(stats["q01"])
    high = np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(low, dtype=bool)))
    normed = np.where(mask, 2.0 * (action - low) / (high - low + 1e-8) - 1.0, action)
    return normed.astype(np.float32)


def unnormalize_action(normed: np.ndarray, stats: dict) -> np.ndarray:
    """Un-normalize action from [-1, 1] back to original scale."""
    low = np.array(stats["q01"])
    high = np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(low, dtype=bool)))
    action = np.where(mask, 0.5 * (normed + 1.0) * (high - low) + low, normed)
    return action.astype(np.float32)
