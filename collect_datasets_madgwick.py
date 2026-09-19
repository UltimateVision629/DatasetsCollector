"""LIBERO collection using raw Joy-Con IMU reports and Madgwick orientation.

This is a parallel collector: ``collect_datasets.py`` remains unchanged.  The
Unity protocol, IK, trajectory format, diagnostics and controls are inherited
unchanged; only Joy-Con roll/pitch/yaw comes from the verified raw-HID
Madgwick path in ``joycon_madgwick.py``.

Run:
    python .\collect_datasets_madgwick.py --ik-backend mujoco

Keep both Joy-Cons still and horizontal during the initial calibration.  The
Joy-Con has no magnetometer, so yaw is intentionally gyro-integrated and may
slowly drift; the horizontal stick remains available for yaw trimming.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque

import numpy as np

import collect_datasets as legacy
from joycon_madgwick import (
    MadgwickImu,
    _G_PER_ACCEL_LSB,
    _RAD_PER_GYRO_LSB,
    _q_conjugate,
    _q_multiply,
    _q_normalize,
    _q_to_rpy,
    _raw_report_samples,
)


class MadgwickJoyConControl:
    """Drop-in ``get_control`` provider with legacy buttons/stick semantics.

    ``JoyconRobotics`` continues to own Bluetooth/HID, stick and button event
    handling.  This class attaches one passive raw-report hook to its gyro HID
    device and never calls its legacy attitude estimator for control.
    """

    _POSITION_SPEED_M_S = 0.10  # preserves old 0.001 m / 10 ms behaviour
    _YAW_TRIM_RAD_S = 1.0       # preserves old 0.01 rad / 10 ms behaviour

    def __init__(self, transport, device: str, beta: float, calibration_seconds: float):
        self.transport = transport
        self.device = device
        self.position = np.array([0.111, 0.0, 0.098], dtype=float)
        self.home_position = self.position.copy()
        self.gripper_open = 1.0
        self.gripper_close = 0.0
        self.gripper_state = self.gripper_open
        self.yaw_trim = 0.0
        self._last_control_time: float | None = None
        self._last_report_time: float | None = None
        self._last_button_state = {"grip": False, "recenter": False}
        self._lock = threading.Lock()
        self._reports: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=4000)
        self._gyro_bias = np.zeros(3)
        self._accel_baseline_norm = 1.0
        self._filter: MadgwickImu | None = None
        self._reference_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._relative_q = self._reference_q.copy()
        self._last_gyro = np.zeros(3)
        self._last_accel = np.zeros(3)
        self._last_accel_ratio = 1.0

        # This hook runs after the existing GyroTrackingJoyCon hook.  It only
        # snapshots raw bytes and does not mutate the legacy estimator state.
        self.transport.gyro.register_update_hook(self._on_raw_report)
        self._calibrate_stationary(calibration_seconds, beta)

    def _on_raw_report(self, gyro_device) -> None:
        try:
            accel_raw, gyro_raw = _raw_report_samples(bytes(gyro_device._input_report))
            with self._lock:
                self._reports.append((time.monotonic(), accel_raw, gyro_raw))
        except (IndexError, ValueError):
            pass

    def _drain_reports(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        with self._lock:
            reports = list(self._reports)
            self._reports.clear()
        return reports

    def _calibrate_stationary(self, seconds: float, beta: float) -> None:
        self._drain_reports()
        print(f"[Madgwick {self.device}] calibrating {seconds:.1f}s; keep it horizontal and still ...")
        end_time = time.monotonic() + seconds
        reports: list[tuple[float, np.ndarray, np.ndarray]] = []
        while time.monotonic() < end_time:
            reports.extend(self._drain_reports())
            time.sleep(0.01)
        if len(reports) < 20:
            raise RuntimeError(f"too few raw IMU reports from {self.device} Joy-Con")

        accel_values = np.asarray([sample[1] for sample in reports])
        gyro_values = np.asarray([sample[2] for sample in reports])
        baseline_accel = np.median(accel_values, axis=0)
        self._gyro_bias = np.median(gyro_values, axis=0)
        self._accel_baseline_norm = max(float(np.linalg.norm(baseline_accel)), 1e-6)
        self._filter = MadgwickImu(beta, baseline_accel)
        self._reference_q = self._filter.q.copy()
        self._relative_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_report_time = None
        print(
            f"[Madgwick {self.device}] raw accel baseline="
            f"{np.array2string(baseline_accel, precision=1)} "
            f"|a|={self._accel_baseline_norm:.1f}; raw gyro bias="
            f"{np.array2string(self._gyro_bias, precision=1)}"
        )

    def _update_orientation(self) -> None:
        assert self._filter is not None
        for timestamp, accel_raw, gyro_raw in self._drain_reports():
            if self._last_report_time is None:
                self._last_report_time = timestamp
                continue
            dt = max(0.001, min(0.050, timestamp - self._last_report_time))
            self._last_report_time = timestamp
            gyro = (gyro_raw - self._gyro_bias) * _RAD_PER_GYRO_LSB
            accel_ratio = float(np.linalg.norm(accel_raw) / self._accel_baseline_norm)
            # Correct with gravity only when acceleration looks quasi-static.
            # This is the key difference from the old filter, which trusted
            # every accelerometer sample even while the operator moved it.
            q = self._filter.update(
                gyro, accel_raw, dt, use_accel=abs(accel_ratio - 1.0) < 0.15)
            self._relative_q = _q_normalize(_q_multiply(q, _q_conjugate(self._reference_q)))
            self._last_gyro = gyro
            self._last_accel = accel_raw * _G_PER_ACCEL_LSB
            self._last_accel_ratio = accel_ratio

    def _step_seconds(self) -> float:
        now = time.monotonic()
        if self._last_control_time is None:
            self._last_control_time = now
            return 0.01
        elapsed = max(0.001, min(0.050, now - self._last_control_time))
        self._last_control_time = now
        return elapsed

    def _button(self, right_name: str, left_name: str) -> bool:
        joycon = self.transport.joycon
        return bool(getattr(joycon, right_name if self.device == "right" else left_name)())

    def _stick(self) -> tuple[int, int]:
        joycon = self.transport.joycon
        if self.device == "right":
            return joycon.get_stick_right_horizontal(), joycon.get_stick_right_vertical()
        return joycon.get_stick_left_horizontal(), joycon.get_stick_left_vertical()

    def _recenter_orientation(self) -> None:
        """Make the current physical pose wrist-neutral without blocking."""
        assert self._filter is not None
        self._reference_q = self._filter.q.copy()
        self._relative_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.yaw_trim = 0.0
        print(f"[Madgwick {self.device}] orientation re-centered.")

    def set_position(self, position) -> None:
        self.position = np.asarray(position, dtype=float).copy()
        self.home_position = self.position.copy()
        self._recenter_orientation()

    def get_control(self):
        self._update_orientation()
        step = self._step_seconds()
        roll, pitch_raw, yaw_imu = _q_to_rpy(self._relative_q)
        # The Madgwick body-frame convention is opposite Unity/SO100's wrist
        # pitch convention.  Map only the published teleop pitch here (not
        # the filter state) so physical forward/back tilt moves the wrist in
        # the expected direction.
        pitch = -pitch_raw
        yaw = yaw_imu - self.yaw_trim
        distance = self._POSITION_SPEED_M_S * step
        stick_h, stick_v = self._stick()

        # Keep the legacy close_y=True movement convention exactly: stick
        # forward/back follows fused pitch/yaw in the x-z plane.
        forward = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ])
        if stick_v > 4000:
            self.position[0] += distance * forward[0]
            self.position[2] += distance * forward[2]
        elif stick_v < 1000:
            self.position[0] -= distance * forward[0]
            self.position[2] -= distance * forward[2]

        if stick_h > 4000:
            self.yaw_trim = min(math.pi / 4, self.yaw_trim + self._YAW_TRIM_RAD_S * step)
            yaw = yaw_imu - self.yaw_trim
        elif stick_h < 1000:
            self.yaw_trim = max(-math.pi / 4, self.yaw_trim - self._YAW_TRIM_RAD_S * step)
            yaw = yaw_imu - self.yaw_trim

        # Shoulder button moves up; stick click moves down; X/B move forward
        # and backward.  These match the existing collector's controls.
        if self._button("get_button_r", "get_button_l"):
            self.position[2] += distance
        if self._button("get_button_r_stick", "get_button_l_stick"):
            self.position[2] -= distance
        if self._button("get_button_x", "get_button_up"):
            self.position[0] += distance
        if self._button("get_button_b", "get_button_down"):
            self.position[0] -= distance

        # Home/capture smoothly returns Cartesian position, like the legacy
        # controller's home button; it does not undo episode records.
        if self._button("get_button_home", "get_button_capture"):
            difference = self.home_position - self.position
            self.position += np.clip(difference, -2.0 * distance, 2.0 * distance)

        grip_pressed = self._button("get_button_zr", "get_button_zl")
        if grip_pressed and not self._last_button_state["grip"]:
            self.gripper_state = (self.gripper_close if self.gripper_state == self.gripper_open
                                  else self.gripper_open)
        self._last_button_state["grip"] = grip_pressed

        recenter_pressed = self._button("get_button_plus", "get_button_minus")
        if recenter_pressed and not self._last_button_state["recenter"]:
            self._recenter_orientation()
            roll = pitch = yaw = 0.0
        self._last_button_state["recenter"] = recenter_pressed

        # A/Y remain episode-level commands, handled by DemoCollector.run().
        button = 0
        if self.device == "right":
            button = 1 if self.transport.joycon.get_button_a() else (-1 if self.transport.joycon.get_button_y() else 0)

        glimit = legacy.CONTROL_GLIMIT
        target = np.array([self.position[0], self.position[1], self.position[2], roll, pitch, yaw])
        target = np.maximum(np.asarray(glimit[0], dtype=float), np.minimum(np.asarray(glimit[1], dtype=float), target))
        self.position[:] = target[:3]
        return target.tolist(), self.gripper_state, button

    def imu_sample(self):
        """Diagnostic payload matching the base collector's CSV columns."""
        return [float(value) for value in (*self._last_gyro, *self._last_accel)]

    def close(self) -> None:
        self.transport.running = False
        for device in (self.transport.joycon, self.transport.gyro, self.transport.button):
            try:
                device._close()
            except (AttributeError, OSError):
                pass


class MadgwickDemoCollector(legacy.DemoCollector):
    """Existing LIBERO collector with only the Joy-Con source replaced."""

    def __init__(self, ik_backend: str, beta: float, calibration_seconds: float):
        # Keep this collector self-contained with respect to the original base
        # collector: it needs no diagnostic/video extensions from the backup.
        super().__init__(ik_backend=ik_backend)
        self.beta = beta
        self.calibration_seconds = calibration_seconds

    def _make_transport(self, device: str):
        return legacy.JoyconRobotics(
            device=device,
            horizontal_stick_mode="yaw_diff",
            close_y=True,
            limit_dof=True,
            glimit=legacy.CONTROL_GLIMIT,
            offset_position_m=list(legacy._SO100_HOME_XYZ),
            common_rad=False,
            lerobot=True,
            pitch_down_double=True,
            without_rest_init=True,
        )

    def init_joycon(self):
        print("Initializing left Joy-Con (raw HID + Madgwick) ...")
        try:
            self.jc_left = MadgwickJoyConControl(
                self._make_transport("left"), "left", self.beta, self.calibration_seconds)
        except (OSError, RuntimeError) as exc:
            print(f"  WARNING: {exc}")
            print("  Left Joy-Con not available; continuing with right only.")
            self.jc_left = None
        else:
            print("  Left Joy-Con ready.")

        print("Initializing right Joy-Con (raw HID + Madgwick) ...")
        try:
            self.jc_right = MadgwickJoyConControl(
                self._make_transport("right"), "right", self.beta, self.calibration_seconds)
        except (OSError, RuntimeError) as exc:
            raise SystemExit(f"ERROR: right Joy-Con initialization failed: {exc}") from exc
        print("  Right Joy-Con ready.")

    def _imu_sample(self, joycon):
        return joycon.imu_sample() if joycon is not None else [""] * 6

    def _go_home(self):
        """Base home motion plus Madgwick position/orientation re-centering."""
        home_target = [legacy._SO100_HOME_XYZ[0], 0.0, legacy._SO100_HOME_XYZ[2], 0.0, 0.0, 0.0]
        for arm_index, controller in ((0, self.jc_right), (1, self.jc_left)):
            if controller is None:
                continue
            self.ik[arm_index].reset_warm()
            if arm_index == 0:
                self.current_arm_q_r = legacy.INIT_ARM_Q.copy()
            else:
                self.current_arm_q_l = legacy.INIT_ARM_Q.copy()
            joints, _ = self._compute_robot_command(arm_index, home_target.copy(), 0.0)
            if joints is not None:
                tag = "robot_0" if arm_index == 0 else "robot_1"
                message = {tag: {"joints": [float(value) for value in joints],
                                 "gripper": 0.0, "button": 0}}
                self.joycon_sock.sendall((legacy.json.dumps(message) + "\n").encode("utf-8"))
                legacy._drain_joycon_feedback(self.joycon_sock)
            controller.set_position(legacy._SO100_HOME_XYZ)
            controller.gripper_state = controller.gripper_open
        time.sleep(0.5)
        print("[Collector] Arms sent to home pose; Madgwick orientation re-centered.")

    def close(self):
        for controller in (getattr(self, "jc_right", None), getattr(self, "jc_left", None)):
            if controller is not None:
                controller.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-backend", default="lerobot", choices=("lerobot", "mujoco", "placo"))
    parser.add_argument("--madgwick-beta", type=float, default=0.04,
                        help="Gravity-correction gain; lower is smoother (default: 0.04)")
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    args = parser.parse_args()

    collector = MadgwickDemoCollector(
        ik_backend=args.ik_backend,
        beta=max(0.0, args.madgwick_beta),
        calibration_seconds=max(1.0, args.calibration_seconds),
    )
    default_language = "pick up the red block"
    try:
        language = input(f"  Language instruction [{default_language}]: ").strip()
    except EOFError:
        language = ""
    collector.language_instruction = language or default_language
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
