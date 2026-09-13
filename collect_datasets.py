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

import gc
import json
import math
import os
import platform
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

COLLECT_FPS = 60.0   # 采集目标帧率（lerobot 式"唯一声明时钟"；--fps 可覆盖）


# ── lerobot 式节拍工具（2026-09-10）──────────────────────────────────────
# precise_sleep 抄自 lerobot `src/lerobot/utils/robot_utils.py:19`（Apache-2.0），
# 参数语义逐字保留。原因：Windows 的 time.sleep 粒度约 15ms，单靠它在 60Hz
# (16.7ms) 节拍上抖动很大 → 观测被零阶保持、帧重复/丢失。lerobot 的做法是
# "睡掉大部分 + 最后 spin_threshold 自旋"，兼顾精度与 CPU。
def precise_sleep(seconds: float, spin_threshold: float = 0.010,
                  sleep_margin: float = 0.005) -> None:
    """等待 seconds，比单独 time.sleep 精确得多（代价是尾部少量自旋）。

    seconds: 要等待的时长
    spin_threshold: 剩余 <= 该值就自旋；否则先 sleep（默认 10ms）
    sleep_margin: sleep 时预留的余量，避免睡过头（默认 5ms）
    """
    if seconds <= 0:
        return
    if platform.system() in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            if remaining > spin_threshold:
                time.sleep(max(remaining - sleep_margin, 0))
            # else: 最后几毫秒自旋（pass）
    else:
        # Linux 的 time.sleep 足够准
        time.sleep(seconds)


def obs_fingerprint(obs: dict) -> bytes:
    """观测指纹：检测"零阶保持"重复帧（状态不变 = 同一帧被重复记录）。

    只取本体感受（关节/末端/夹爪）。诊断已确认状态重复与图像重复 100% 一致，
    故用状态当整帧代理，避免每帧做图像哈希的开销。
    """
    parts = []
    for k in ("robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos"):
        v = obs.get(k)
        if v is not None:
            parts.append(np.asarray(v, dtype=np.float64).tobytes())
    return b"".join(parts)


class LoopTiming:
    """lerobot 式节拍诊断：实际频率 / 超时占比 / obs 重复率。

    lerobot 的 record_loop 跑不赢目标 FPS 时会告警（"Dataset frames might be
    dropped"）。本类把这件事**量化**：限频告警 + 每集数值落盘，让"两个时钟域"
    从静默的 ~46% 重复变成可读数字。
    """

    def __init__(self, target_fps: float = COLLECT_FPS,
                 report_every_s: float = 2.0, warn_every_s: float = 5.0):
        self.target_fps = float(target_fps)
        self.control_interval = 1.0 / float(target_fps)
        self.report_every_s = report_every_s
        self.warn_every_s = warn_every_s
        self._reset_window()
        self._reset_episode()
        self._last_fp = None
        self._last_warn = 0.0

    def _reset_window(self) -> None:
        self._win_start = time.perf_counter()
        self._win_ticks = 0
        self._win_sum_dt = 0.0
        self._win_over = 0
        self._win_max_dt = 0.0
        self._win_obs = 0
        self._win_dup = 0

    def _reset_episode(self) -> None:
        self._ep_start = time.perf_counter()
        self._ep_ticks = 0
        self._ep_over = 0
        self._ep_obs = 0
        self._ep_dup = 0

    def note_tick(self, dt_s: float) -> None:
        """记一次循环耗时（dt_s = 本轮实际耗时）。"""
        self._win_ticks += 1
        self._win_sum_dt += dt_s
        self._ep_ticks += 1
        if dt_s > self.control_interval:
            self._win_over += 1
            self._ep_over += 1
            if dt_s > self._win_max_dt:
                self._win_max_dt = dt_s

    def note_obs(self, obs: dict) -> None:
        """记一次观测，并统计是否与上一次完全相同（零阶保持重复）。"""
        fp = obs_fingerprint(obs)
        self._win_obs += 1
        self._ep_obs += 1
        if self._last_fp is not None and fp == self._last_fp:
            self._win_dup += 1
            self._ep_dup += 1
        self._last_fp = fp

    def maybe_report(self) -> None:
        """每 report_every_s 打一行实际节拍；超时则限频告警。"""
        now = time.perf_counter()
        el = now - self._win_start
        if el < self.report_every_s or self._win_ticks == 0:
            return
        hz = self._win_ticks / el
        over_pct = 100.0 * self._win_over / self._win_ticks
        dup_pct = 100.0 * self._win_dup / max(1, self._win_obs)
        mean_dt = self._win_sum_dt / self._win_ticks
        print(f"[Timing] 目标 {self.target_fps:.0f}Hz | 实际 {hz:.1f}Hz | "
              f"超时 {over_pct:.1f}% (最坏 {self._win_max_dt * 1000:.1f}ms) | "
              f"平均dt {mean_dt * 1000:.1f}ms | obs重复 {dup_pct:.1f}%")
        if over_pct > 5.0 and now - self._last_warn > self.warn_every_s:
            self._last_warn = now
            print("[Timing] ⚠️ 循环跑不赢目标 FPS —— 观测会被零阶保持、帧会重复/丢弃。"
                  "常见原因: 1) get_obs 的 TCP 往返 + Unity 侧 Render/ReadPixels/PNG 编码"
                  " 2) 每帧 IK 开销 3) Windows time.sleep 粒度(~15ms)。"
                  "对照 lerobot 结论：优先把观测生产移出请求-响应")
        self._reset_window()

    def episode_summary(self) -> dict:
        """取本集统计并清零（供 save_episode 落盘）。"""
        el = max(1e-9, time.perf_counter() - self._ep_start)
        s = {
            "loop_fps_target": round(self.target_fps, 3),
            "loop_fps_actual": round(self._ep_ticks / el, 3),
            "loop_overrun_pct": round(100.0 * self._ep_over / max(1, self._ep_ticks), 2),
            "loop_obs_dup_pct": round(100.0 * self._ep_dup / max(1, self._ep_obs), 2),
            "loop_ticks": int(self._ep_ticks),
        }
        self._reset_episode()
        self._last_fp = None
        return s


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

    def __init__(self, ik_backend: str = "lerobot", ik_tol: float = 1e-3,
                 fps: float = COLLECT_FPS, apply_delay: float = 0.01):
        self.joycon_sock: socket.socket = None
        self.env: LiberoUnityEnv = None
        # ONE IK INSTANCE PER ARM (MuJoCoIK tracks a per-arm so100 warm)
        self.ik = [create_ik(ik_backend),
                   create_ik(ik_backend)]
        self.ik_tol = ik_tol   # mujoco 后端 least_squares 容差（--ik-tol）
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
        # ── lerobot 式单一时钟（2026-09-10）──
        self.fps = float(fps)
        self.control_interval = 1.0 / self.fps
        # 兼容旧行为：发完关节后给 Unity 一点时间应用（lerobot 无此延迟）。
        # 0.01s 占 60Hz 预算(16.7ms)的 60%，是提速的主要杠杆 → --apply-delay 0 可关。
        self.apply_delay = float(apply_delay)
        self._timing = LoopTiming(self.fps)
        self.episode_loop_dt: list = []   # 每帧实际循环耗时（落盘，供离线审计）

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

        # tol: mujoco 后端 least_squares 容差——采集是闭环的（操作员视觉补偿），
        # 放宽到 1e-3 让 IK 快数倍（--ik-tol 可调）；推理/回放保持默认 1e-8。
        joints5, new_q = self.ik[arm_index].solve(
            target_pose, gripper_state, current_arm_q, tol=self.ik_tol)
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
        # lerobot 式：本集节拍统计清零（上一集的数字已在 save_episode 落盘）
        self._timing._reset_episode()
        self._timing._last_fp = None
        self.episode_loop_dt = []

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

        # Eye-in-hand image（新场景 libero_put_block_in_box 有该键；旧场景无 → 不存）
        if "eye_in_hand_image" in obs_list[0]:
            data["eye_in_hand_image"] = np.stack([o["eye_in_hand_image"] for o in obs_list])

        # ── lerobot 式时间基（2026-09-10）──────────────────────────────
        # lerobot 的约定：调用方**不得**自带 timestamp/frame_index，由 dataset 按
        # `frame_index / fps` 推导（dataset_writer.py:180）——时间轴 = 帧计数器，
        # 不是墙钟。这里在原始 npz 里也落一份，让采集数据自描述、可离线审计。
        n = len(self.episode_actions)
        data["frame_index"] = np.arange(n, dtype=np.int64)
        data["timestamp"] = np.arange(n, dtype=np.float64) / self.fps

        # 本集节拍诊断：实际 Hz / 超时占比 / obs 重复率（lerobot 只告警，这里落盘）
        timing = self._timing.episode_summary()
        data.update(timing)

        # 每帧实际循环耗时（与 action 等长时落盘；长度不符则**告警**并跳过，不静默错位）
        if len(self.episode_loop_dt) == n:
            data["loop_dt"] = np.asarray(self.episode_loop_dt, dtype=np.float64)
        else:
            print(f"[Collector] ⚠️ loop_dt 长度 {len(self.episode_loop_dt)} ≠ action 数 {n}，"
                  f"本集不落盘 loop_dt（节拍统计仍有效）——请检查记录顺序")

        filepath = os.path.join(OUTPUT_DIR, filename)
        np.savez_compressed(filepath, **data)
        print(f"[Collector] Saved: {filepath}  ({n} steps, success={success})")
        print(f"[Collector] 节拍: 目标 {timing['loop_fps_target']:.0f}Hz | "
              f"实际 {timing['loop_fps_actual']:.1f}Hz | "
              f"超时 {timing['loop_overrun_pct']:.1f}% | "
              f"obs重复 {timing['loop_obs_dup_pct']:.1f}%")

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self):
        # 禁用 GC（2026-09-02，实测）：scipy least_squares 每帧分配的小对象
        # 触发 gen1/gen2 收集暂停 10-16ms——采集偶发"卡一下/手跟不上"的根因
        # （benchmark_ik_perf.py: gc off 后卡帧 max 16.5→5.4ms）。本脚本对象
        # 全部由引用计数即时释放（无循环引用），disable 安全。
        gc.disable()
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
        # ── lerobot 式单一时钟（2026-09-10）───────────────────────────
        # lerobot 的 record_loop 用"唯一声明 fps + 自旋精度睡眠 + 跑不赢就告警"，
        # 且数据集时间轴 = frame_index/fps（不是墙钟）。这里照搬该纪律：不再用
        # `time.sleep(dt - elapsed)`（Windows 粒度 ~15ms，锁不住 60Hz 节拍）。
        control_interval = self.control_interval
        print(f"  [Clock] 目标 {self.fps:.0f}Hz (interval {control_interval * 1000:.1f}ms) | "
              f"apply_delay {self.apply_delay * 1000:.0f}ms | "
              f"单一时钟域 = 本循环节拍")

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
                    # 两臂 IK 全失败：仍要走完整节拍（维持单一时钟，不空转）
                    dt_s = time.perf_counter() - loop_start
                    self._timing.note_tick(dt_s)
                    precise_sleep(max(control_interval - dt_s, 0.0))
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
                if self.apply_delay > 0:
                    # 旧行为：给 Unity 时间应用关节。lerobot 无此延迟 —— 它占
                    # 60Hz 预算(16.7ms)的 60%，是提速的首要杠杆（--apply-delay 0 关闭）。
                    time.sleep(self.apply_delay)

                # ── Step 5: Get observation from TrainingServer ──────────
                obs = self.env.get_obs()
                self._timing.note_obs(obs)   # 统计零阶保持重复帧

                # ── Step 6: Record (label = sender-side EEF delta) ──────
                delta_action = np.concatenate([action_r, action_l])  # 14-dim
                self.record_step(obs, delta_action)

                # 本 tick 耗时（不含节拍睡眠）。**必须在 Step 7 之前记**：按钮触发
                # save_episode 时会读 episode_loop_dt，晚记会让它比 action 少一项
                # → 长度守卫判为不符 → loop_dt 永远存不上。
                dt_s = time.perf_counter() - loop_start
                self.episode_loop_dt.append(dt_s)

                # ── Step 7: Handle button events (edge-triggered) ───────
                if button != prev_button:
                    if button == 1:  # A button → save success, start new episode (auto home)
                        self.save_episode(success=True)
                        self.start_episode()
                    elif button == -1:  # Y button → discard, start new episode (auto home)
                        print("[Collector] Episode discarded.")
                        self.start_episode()
                prev_button = button

                # ── Step 8: 固定节拍（lerobot 方案）─────────────────────
                self._timing.note_tick(dt_s)
                self._timing.maybe_report()      # 实际Hz/超时/重复率 + 限频告警
                precise_sleep(max(control_interval - dt_s, 0.0))

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
    parser.add_argument("--ik-tol", type=float, default=1e-3,
                        help="mujoco 后端 least_squares 容差（ftol/xtol）。采集是闭环的，"
                             "1e-3 已足够且快数倍；1e-8 = 最高精度（回放/推理默认）。"
                             "仅 --ik-backend mujoco 时生效")
    parser.add_argument("--fps", type=float, default=COLLECT_FPS,
                        help=f"采集目标帧率（lerobot 式唯一声明时钟，默认 {COLLECT_FPS:.0f}）。"
                             "跑不赢会告警并显示实际 Hz；不要随意更改——它决定数据的"
                             "时间语义（现有模型按 ~60Hz 训练）")
    parser.add_argument("--apply-delay", type=float, default=0.01,
                        help="发完关节后等待 Unity 应用的秒数（旧行为 0.01，保留以兼容）。"
                             "lerobot 无此延迟；设 0 可回收 60Hz 预算的 ~60%%，是提速首要杠杆")
    args = parser.parse_args()

    collector = DemoCollector(ik_backend=args.ik_backend, ik_tol=args.ik_tol,
                              fps=args.fps, apply_delay=args.apply_delay)

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
