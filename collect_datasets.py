"""
Demo collection script for LIBERO block grasping.

Uses Joy-Con teleop (via joyconrobotics) to control the robot in Unity,
while recording (observation, action) pairs from TrainingServer.

Architecture (LeRobot-style processor separation):
  Joy-Con get_control() → target_pose, gripper, button
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   teleop_action_processor          robot_action_processor
   (target_pose → EEF delta)       (target_pose → IK → joints)
              │                               │
              ▼                               ▼
        action_label                    send_joints() → Unity
        (saved to .npz)                (JoyConReceiver:5555)

Key difference from old approach: action labels are SENDER-side
(Joy-Con target deltas), NOT receiver-side (Unity observation deltas).
This matches LeRobot's philosophy: labels = operator intent, commands = robot execution.

Workflow:
  1. Start Unity Play mode (starts both JoyConReceiver:5555 and TrainingServer:5556)
  2. Run this script
  3. Use Joy-Con to perform the task
  4. Press A button → save trajectory as successful episode
  5. Press Y button → discard and restart episode

Output: trajectories saved to ../network/demos/ as .npz files
Action format (matching VLA-Adapter LIBERO OSC_POSE):
  - Per arm: [dx, dy, dz, dRx, dRy, dRz, gripper] (7-dim EEF delta)
  - Dual-arm: concatenated [arm0:7, arm1:7] = 14-dim
  - dPos: EEF position delta (meters)
  - dRot: axis-angle rotation delta (axis * angle in radians)
  - gripper: absolute gripper target (0=open, 1=closed)
"""

import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# ── Joy-Con support ────────────────────────────────────────────────────
try:
    from joyconrobotics import JoyconRobotics
except ImportError:
    print("ERROR: joyconrobotics not installed.")
    print("  pip install joyconrobotics")
    sys.exit(1)

# Local import — env_client.py lives in the same directory
from env_client import LiberoUnityEnv

# Unified IK backends (arm_ik.py in libero-unity/test) — collect/replay/
# inference share the same pluggable IK.  Default keeps the original
# lerobot_IK behavior; use --ik-backend mujoco for MuJoCo-accurate joints.
_TEST = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "libero-unity", "test"))
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)
from arm_ik import create_ik, INIT_ARM_Q, CONTROL_GLIMIT  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────
JOYCON_PORT = 5555   # JoyConReceiver
TRAIN_PORT = 5556    # TrainingServer
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demos")

_SO100_HOME_XYZ = [0.111, 0.0, 0.098]


# ── Quaternion math (Unity convention: [x, y, z, w]) ────────────────────

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion [x, y, z, w]."""
    c = q.copy()
    c[:3] *= -1.0
    return c

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (both [x, y, z, w])."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to axis-angle vector (axis * angle)."""
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z)
    if norm < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(norm, max(min(w, 1.0), -1.0))
    axis = np.array([x, y, z]) / norm
    return axis * angle

def compute_eef_delta(curr_pos, curr_quat, prev_pos, prev_quat, gripper) -> np.ndarray:
    """Compute 7-dim EEF delta action: [dx, dy, dz, dRx, dRy, dRz, gripper].

    Position delta: simple subtraction.
    Rotation delta: relative quaternion → axis-angle vector.
    Gripper: absolute target value (0=open, 1=closed).

    NOTE: This function computes RECEIVER-side deltas (from observations).
    For sender-side labels, use _compute_action_label_from_target() in DemoCollector.
    """
    dpos = curr_pos - prev_pos
    q_rel = quat_multiply(curr_quat, quat_conjugate(prev_quat))
    drot = quat_to_rotvec(q_rel)
    return np.array([
        dpos[0], dpos[1], dpos[2],
        drot[0], drot[1], drot[2],
        float(gripper),
    ], dtype=np.float32)


def _euler_zxy_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert extrinsic Z-X-Y Euler angles to quaternion [x, y, z, w].

    Matches Unity's Quaternion.Euler(rx, ry, rz) which applies
    Rz(rz) first, then Rx(rx), then Ry(ry) — extrinsic Z-X-Y.

    Rotation matrix: R = Ry(ry) * Rx(rx) * Rz(rz)
    Quaternion: q = q_y(ry) ⊗ q_x(rx) ⊗ q_z(rz)

    Args:
        rx: rotation around X axis (radians) — pitch in Unity convention
        ry: rotation around Y axis (radians) — yaw in Unity convention
        rz: rotation around Z axis (radians) — roll in Unity convention

    Returns:
        Unit quaternion [x, y, z, w]
    """
    hx, hy, hz = rx / 2.0, ry / 2.0, rz / 2.0
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    # q = q_y ⊗ q_x ⊗ q_z  (extrinsic Z-X-Y)
    return np.array([
        cy * sx * cz + sy * cx * sz,   # x
        sy * cx * cz - cy * sx * sz,   # y
        cy * cx * sz - sy * sx * cz,   # z
        cy * cx * cz + sy * sx * sz,   # w
    ])


def _drain_joycon_feedback(sock: socket.socket) -> None:
    """Discard Unity's joint-feedback replies so its write never blocks.

    JoyConReceiver writes one feedback JSON line back per received message
    (JoyConReceiver.cs WriteJointFeedback); nothing on this side reads it.
    Without draining, the OS receive buffer fills after a few minutes of
    continuous teleop, Unity's recv thread blocks inside stream.Write and
    stops reading → the arm freezes at the last pose and Python's sendall
    later blocks too (the "freeze ~7 episodes into a batch" bug — time-based,
    independent of which files are replayed).
    """
    sock.setblocking(False)
    try:
        while True:
            try:
                if not sock.recv(65536):
                    break  # peer closed
            except (BlockingIOError, InterruptedError):
                break  # buffer drained
    finally:
        sock.setblocking(True)


class DemoCollector:
    """Collects demonstrations using Joy-Con teleop + Unity TrainingServer."""

    def __init__(self, ik_backend: str = "lerobot"):
        self.joycon_sock: socket.socket = None
        self.env: LiberoUnityEnv = None
        # ONE IK INSTANCE PER ARM (MuJoCoIK tracks a per-arm so100 warm)
        self.ik = [create_ik(ik_backend),
                   create_ik(ik_backend)]
        self.current_arm_q_r = INIT_ARM_Q.copy()
        self.current_arm_q_l = INIT_ARM_Q.copy()
        self.prev_joint_angles_0: np.ndarray = None
        self.prev_joint_angles_1: np.ndarray = None
        self.episode_obs: list = []
        self.episode_actions: list = []
        self.episode_count = 0
        # Joy-Con target tracking for sender-side EEF delta computation
        # _prev_jc_state[arm_index] = None (first frame) or dict with
        #   'world_pos', 'world_quat', 'gripper'
        self._prev_jc_state = [None, None]

    # ── Joy-Con setup ──────────────────────────────────────────────────

    def init_joycon(self):
        offset = list(_SO100_HOME_XYZ)

        print("Initializing left Joy-Con ...")
        try:
            self.jc_left = JoyconRobotics(
                device="left",
                horizontal_stick_mode="yaw_diff",
                close_y=True,
                limit_dof=True,
                glimit=CONTROL_GLIMIT,
                offset_position_m=offset,
                common_rad=False,
                lerobot=True,
                pitch_down_double=True,
            )
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            print("  Left Joy-Con not available, continuing with right only.")
            self.jc_left = None
        else:
            print("  Left Joy-Con ready.")

        print("Initializing right Joy-Con ...")
        try:
            self.jc_right = JoyconRobotics(
                device="right",
                horizontal_stick_mode="yaw_diff",
                close_y=True,
                limit_dof=True,
                glimit=CONTROL_GLIMIT,
                offset_position_m=offset,
                common_rad=False,
                lerobot=True,
                pitch_down_double=True,
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
        print("  Right Joy-Con ready.")

    # ── Networking ─────────────────────────────────────────────────────

    def connect_joycon_server(self):
        """Connect to Unity JoyConReceiver on port 5555."""
        print(f"Connecting to JoyConReceiver on 127.0.0.1:{JOYCON_PORT} ...")
        self.joycon_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.joycon_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                self.joycon_sock.connect(("127.0.0.1", JOYCON_PORT))
                break
            except (ConnectionRefusedError, OSError):
                print("  Waiting for Unity JoyConReceiver ...")
                time.sleep(1.0)
        print("  Connected.")

    def connect_env(self):
        """Connect to Unity TrainingServer on port 5556."""
        self.env = LiberoUnityEnv(port=TRAIN_PORT)
        self.env.connect()

    def send_joints_to_unity(self, joints_r: np.ndarray, gripper_r: float,
                              joints_l: np.ndarray = None, gripper_l: float = 1.0):
        """Send joint angles for both arms to Unity via JoyConReceiver."""
        msg = {
            "robot_0": {
                "joints": [float(v) for v in joints_r],
                "gripper": float(gripper_r),
                "button": 0,
            }
        }
        if joints_l is not None:
            msg["robot_1"] = {
                "joints": [float(v) for v in joints_l],
                "gripper": float(gripper_l),
                "button": 0,
            }
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self.joycon_sock.sendall(data)
        _drain_joycon_feedback(self.joycon_sock)

    # ── Processors (LeRobot-style: label ← teleop action processor, command ← robot action processor) ─

    def _go_home(self):
        """Send both arms to the naturally-bent ready pose via JoyConReceiver.

        Resets IK warm-starts to _INIT_ARM_Q, runs IK for a home target pose,
        sends joints to Unity, and resets Joy-Con internal position accumulators
        so the next get_control() returns values near home (prevents snap-back).
        """
        home_target = [_SO100_HOME_XYZ[0], 0.0, _SO100_HOME_XYZ[2], 0.0, 0.0, 0.0]

        for arm_idx, jc in [(0, self.jc_right), (1, self.jc_left)]:
            if jc is None:
                continue

            # Reset IK warm-start so we converge to home, not current pose
            self.ik[arm_idx].reset_warm()  # per-episode warm reset
            if arm_idx == 0:
                self.current_arm_q_r = INIT_ARM_Q.copy()
            else:
                self.current_arm_q_l = INIT_ARM_Q.copy()

            joints, _ = self._compute_robot_command(arm_idx, home_target.copy(), 0.0)
            if joints is not None:
                tag = "robot_0" if arm_idx == 0 else "robot_1"
                msg = {
                    tag: {
                        "joints": [float(v) for v in joints],
                        "gripper": 0.0,
                        "button": 0,
                    }
                }
                data = (json.dumps(msg) + "\n").encode("utf-8")
                self.joycon_sock.sendall(data)
                _drain_joycon_feedback(self.joycon_sock)

            # Reset Joy-Con internal position accumulator to home offset.
            # Without this, get_control() returns the pre-reset position and
            # the robot snaps back to the old pose on the next loop iteration.
            jc.set_position(list(_SO100_HOME_XYZ))

            # Reset accumulated yaw differential (horizontal stick yaw_diff mode).
            # Without this, accumulated yaw offset persists across resets.
            jc.yaw_diff = 0.0
            jc.orientation_sensor.set_yaw_diff(0.0)

            # Reset the fused orientation vectors (roll/pitch/yaw) so the arm
            # returns to home with a neutral wrist.  roll/pitch are gyro-
            # integrated and would otherwise keep the tilt held at Y-press
            # time (joyconrobotics' own reset uses reset_yaw + set_yaw_diff).
            jc.orientation_sensor.reset_yaw()

            # Reset gripper state to open, matching the home pose.
            jc.gripper_state = jc.gripper_open

        # Wait for arms to settle into home pose
        time.sleep(0.5)
        print("[Collector] Arms sent to home pose.")

    def _compute_robot_command(self, arm_index: int, target_pose: list, gripper_state: float):
        """Robot action processor: Joy-Con target → IK → joint angles.

        Takes a pre-read Joy-Con target (to avoid double-reading the device)
        and computes joint angles via lerobot_IK.

        Args:
            arm_index: 0 = right arm (jc_right), 1 = left arm (jc_left)
            target_pose: [x_r, _, z_r, roll_r, pitch_r, yaw_r] from Joy-Con
            gripper_state: raw gripper value from Joy-Con

        Returns:
            (joint_angles_6d, gripper, button) or (None, gripper, button) on IK failure
        """
        # Delegate to the pluggable IK backend (arm_ik.py).
        # Gripper stays RAW (0/1) — JoyConReceiver SetJoint() passes it
        # straight to the Jaw actuator (historical collect behavior, 勿改).
        current_arm_q = self.current_arm_q_r if arm_index == 0 else self.current_arm_q_l

        joints5, new_q = self.ik[arm_index].solve(target_pose, gripper_state, current_arm_q)
        if joints5 is None:
            return None, gripper_state

        target_qpos = np.concatenate((joints5, [gripper_state]))
        # Update current arm q for next IK call
        if arm_index == 0:
            self.current_arm_q_r = new_q
        else:
            self.current_arm_q_l = new_q
        return target_qpos, gripper_state

    def _compute_action_label(self, arm_index: int, target_pose: list, gripper_state: float) -> np.ndarray:
        """Teleop action processor: Joy-Con target → 7-dim EEF delta.

        Computes the sender-side EEF delta action from consecutive Joy-Con
        target poses in world-frame Cartesian coordinates. This is the
        operator's intended motion — the action label for training.

        The Joy-Con target_pose is in a LOCAL cylindrical frame that rotates
        with the base yaw. We convert to base-relative Cartesian coords,
        apply the base yaw rotation, and compute frame-to-frame deltas.

        First frame: stores initial world-frame pose, returns zero delta.

        Args:
            arm_index: 0 = right arm, 1 = left arm
            target_pose: [x_r, _, z_r, roll_r, pitch_r, yaw_r] from Joy-Con
            gripper_state: raw gripper value from Joy-Con

        Returns:
            7-dim np.ndarray: [dx, dy, dz, dRx, dRy, dRz, gripper]
        """
        # Clamp to workspace limits (same as _compute_robot_command)
        for i in range(6):
            target_pose[i] = max(CONTROL_GLIMIT[0][i], min(CONTROL_GLIMIT[1][i], target_pose[i]))

        x_r, _, z_r, roll_r, pitch_r, yaw_r = target_pose
        y_r = 0.01  # fixed lateral offset

        # Same orientation transforms as _compute_robot_command
        pitch_r = -pitch_r
        roll_r = roll_r - math.pi / 2

        # World-frame position: rotate base-relative [x_r, y_r, z_r] by base yaw
        cos_y, sin_y = math.cos(yaw_r), math.sin(yaw_r)
        world_x = x_r * cos_y - y_r * sin_y
        world_y = x_r * sin_y + y_r * cos_y
        world_z = z_r
        world_pos = np.array([world_x, world_y, world_z])

        # World-frame orientation: base yaw quat * wrist quat
        # Wrist orientation matches Unity Quaternion.Euler(jcPitch, 0, -jcRoll)
        # After our transforms: pitch_r = -raw_pitch, roll_r = raw_roll - π/2
        # In raw terms: Unity X = raw_pitch = -pitch_r, Z = -raw_roll = -(roll_r + π/2)
        q_wrist = _euler_zxy_to_quat(-pitch_r, 0.0, -(roll_r + math.pi / 2))
        q_base = _euler_zxy_to_quat(0.0, 0.0, yaw_r)
        world_quat = quat_multiply(q_base, q_wrist)

        # First frame for this arm: store state, return zero delta
        prev = self._prev_jc_state[arm_index]
        if prev is None:
            self._prev_jc_state[arm_index] = {
                'world_pos': world_pos.copy(),
                'world_quat': world_quat.copy(),
                'gripper': gripper_state,
            }
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper_state)], dtype=np.float32)

        # Compute deltas from previous frame
        dpos = world_pos - prev['world_pos']
        q_rel = quat_multiply(world_quat, quat_conjugate(prev['world_quat']))
        drot = quat_to_rotvec(q_rel)

        # Update state
        self._prev_jc_state[arm_index] = {
            'world_pos': world_pos.copy(),
            'world_quat': world_quat.copy(),
            'gripper': gripper_state,
        }

        return np.array([
            dpos[0], dpos[1], dpos[2],
            drot[0], drot[1], drot[2],
            float(gripper_state),
        ], dtype=np.float32)

    # ── Episode management ─────────────────────────────────────────────

    def start_episode(self):
        """Reset Unity, move arms to home pose, then begin recording."""
        print("\n[Collector] Starting new episode...")
        self.env.reset()
        self._go_home()
        time.sleep(0.5)  # let arms settle into home pose
        obs = self.env.get_obs()
        # Reset Joy-Con target tracking (first frame = no delta)
        self._prev_jc_state = [None, None]
        self.episode_obs = [obs]
        self.episode_actions = []

    def record_step(self, obs, action):
        self.episode_obs.append(obs)
        self.episode_actions.append(action)

    def save_episode(self, success: bool):
        """Save trajectory to disk."""
        if len(self.episode_actions) == 0:
            print("[Collector] Empty episode, skipping save.")
            return

        lang = self.language_instruction

        # Sanitize for filename: lowercase, replace spaces/slashes with underscores
        lang_slug = lang.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove any other non-filename-safe characters
        lang_slug = "".join(c for c in lang_slug if c.isalnum() or c == "_").strip("_") or "task"

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_count += 1
        filename = f"episode_{self.episode_count:04d}_{lang_slug}_{ts}_{'success' if success else 'partial'}.npz"

        # Stack observations (skip last obs since we have N actions and N+1 observations)
        obs_list = self.episode_obs[:len(self.episode_actions)]

        data = {
            "agentview_image": np.stack([o["agentview_image"] for o in obs_list]),
            "robot0_joint_pos": np.stack([o["robot0_joint_pos"] for o in obs_list]),
            "robot0_eef_pos": np.stack([o["robot0_eef_pos"] for o in obs_list]),
            "robot0_eef_quat": np.stack([o["robot0_eef_quat"] for o in obs_list]),
            "robot0_gripper_qpos": np.stack([o["robot0_gripper_qpos"] for o in obs_list]),
            "robot1_joint_pos": np.stack([o["robot1_joint_pos"] for o in obs_list]),
            "robot1_eef_pos": np.stack([o["robot1_eef_pos"] for o in obs_list]),
            "robot1_eef_quat": np.stack([o["robot1_eef_quat"] for o in obs_list]),
            "robot1_gripper_qpos": np.stack([o["robot1_gripper_qpos"] for o in obs_list]),
            "action": np.stack(self.episode_actions),
            "language_instruction": lang,
            "success": success,
        }

        filepath = os.path.join(OUTPUT_DIR, filename)
        np.savez_compressed(filepath, **data)
        print(f"[Collector] Saved: {filepath}  ({len(self.episode_actions)} steps, success={success})")

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 60)
        print("  LIBERO Dual-Arm Demo Collector (LeRobot-style)")
        print("  Controls:")
        print("    Right Joy-Con  → right arm (robot_0)")
        print("    Left Joy-Con   → left arm (robot_1)")
        print("    ZL / ZR        → toggle gripper open/close")
        print("    A button (R)   → save episode as SUCCESS")
        print("    Y button (R)   → discard & restart episode")
        print("=" * 60 + "\n")

        self.start_episode()
        prev_button = 0
        dt = 0.016

        while True:
            try:
                loop_start = time.perf_counter()

                # ── Step 1: Read Joy-Cons (once per arm per frame) ──────
                # Right arm
                target_r, gripper_raw_r, btn_r = self.jc_right.get_control()
                button = btn_r  # right JoyCon buttons for episode control

                target_l, gripper_raw_l, btn_l = None, 1.0, 0
                if self.jc_left is not None:
                    target_l, gripper_raw_l, btn_l = self.jc_left.get_control()

                # ── Step 2: Teleop action processor → EEF delta labels ──
                action_r = self._compute_action_label(0, target_r, gripper_raw_r)
                if target_l is not None:
                    action_l = self._compute_action_label(1, target_l, gripper_raw_l)
                else:
                    action_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

                # ── Step 3: Robot action processor → IK → joint angles ──
                joints_r, gripper_r = self._compute_robot_command(0, target_r, gripper_raw_r)
                joints_l, gripper_l = None, 1.0
                if target_l is not None:
                    joints_l, gripper_l = self._compute_robot_command(1, target_l, gripper_raw_l)

                # Skip frame only if BOTH arms failed IK
                if joints_r is None and joints_l is None:
                    time.sleep(dt)
                    continue

                # ── Step 4: Send joint angles to Unity (JoyConReceiver) ──
                msg = {}
                if joints_r is not None:
                    msg["robot_0"] = {
                        "joints": [float(v) for v in joints_r],
                        "gripper": float(gripper_r),
                        "button": 0,
                    }
                if joints_l is not None:
                    msg["robot_1"] = {
                        "joints": [float(v) for v in joints_l],
                        "gripper": float(gripper_l),
                        "button": 0,
                    }
                if msg:
                    data = (json.dumps(msg) + "\n").encode("utf-8")
                    self.joycon_sock.sendall(data)
                    _drain_joycon_feedback(self.joycon_sock)
                time.sleep(0.01)  # give Unity time to apply

                # ── Step 5: Get observation from TrainingServer ──────────
                obs = self.env.get_obs()

                # ── Step 6: Record (label = sender-side EEF delta) ──────
                delta_action = np.concatenate([action_r, action_l])  # 14-dim
                self.record_step(obs, delta_action)

                # ── Step 7: Handle button events (edge-triggered) ───────
                if button != prev_button:
                    if button == 1:  # A button → save success, start new episode (auto home)
                        self.save_episode(success=True)
                        self.start_episode()
                    elif button == -1:  # Y button → discard, start new episode (auto home)
                        print("[Collector] Episode discarded.")
                        self.start_episode()
                prev_button = button

                # ── Step 8: Timing ──────────────────────────────────────
                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            except KeyboardInterrupt:
                print("\n[Collector] Interrupted.")
                break
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

    def close(self):
        if self.joycon_sock:
            self.joycon_sock.close()
        if self.env:
            self.env.close()
        print("[Collector] Shut down.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-backend", default="lerobot",
                        choices=["lerobot", "mujoco", "placo"],
                        help="IK backend: lerobot (default, original behavior), "
                             "mujoco (MuJoCo-accurate), placo (stub)")
    args = parser.parse_args()

    collector = DemoCollector(ik_backend=args.ik_backend)

    # ── Language instruction: entered on the command line (not Unity) ──
    # Prompt FIRST so the operator has time to focus Unity while the script
    # connects / returns home; blank keeps the default.
    default_lang = "pick up the red block"
    try:
        lang = input(f"  Language instruction [{default_lang}]: ").strip()
    except EOFError:
        lang = ""
    collector.language_instruction = lang or default_lang
    print(f"  Instruction: {collector.language_instruction}")

    try:
        collector.init_joycon()
        collector.connect_joycon_server()
        collector.connect_env()
        collector.run()
    except KeyboardInterrupt:
        pass
    finally:
        collector.close()


if __name__ == "__main__":
    main()
