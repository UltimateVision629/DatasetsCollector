"""Dependency-free raw Joy-Con IMU helpers for Madgwick teleoperation.

The module deliberately contains no HID connection, GUI, video, or CSV logic.
``collect_datasets_madgwick.py`` owns those concerns and uses these functions
to bypass the legacy Joy-Con attitude estimator.
"""
from __future__ import annotations

import math

import numpy as np


# Joy-Con's default IMU ranges are ±2000 dps and ±8 g.
_RAD_PER_GYRO_LSB = math.radians(4000.0 / 65535.0)
_G_PER_ACCEL_LSB = 16000.0 / 65535.0 / 1000.0


def _int16_le(low: int, high: int) -> int:
    value = low | (high << 8)
    return value if value < 0x8000 else value - 0x10000


def _raw_report_samples(report: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode and average the three IMU samples in a Joy-Con 0x30 report."""
    if len(report) < 49 or report[0] != 0x30:
        raise ValueError("expected a complete Joy-Con 0x30 IMU report")
    accel: list[list[float]] = []
    gyro: list[list[float]] = []
    for sample in range(3):
        offset = 13 + sample * 12
        accel.append([_int16_le(report[offset], report[offset + 1]),
                      _int16_le(report[offset + 2], report[offset + 3]),
                      _int16_le(report[offset + 4], report[offset + 5])])
        gyro.append([_int16_le(report[offset + 6], report[offset + 7]),
                     _int16_le(report[offset + 8], report[offset + 9]),
                     _int16_le(report[offset + 10], report[offset + 11])])
    return np.mean(accel, axis=0), np.mean(gyro, axis=0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3)


def _q_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _q_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _q_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion rotating one unit vector onto another."""
    source, target = _normalize(source), _normalize(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -0.999999:
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            basis = np.array([0.0, 1.0, 0.0])
        return np.array([0.0, *_normalize(np.cross(source, basis))])
    return _q_normalize(np.array([1.0 + dot, *np.cross(source, target)]))


def _q_to_rpy(q: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ roll/pitch/yaw in radians from a quaternion."""
    w, x, y, z = _q_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class MadgwickImu:
    """Madgwick's IMU-only quaternion update (gravity corrects roll/pitch)."""

    def __init__(self, beta: float, initial_accel: np.ndarray):
        self.beta = float(beta)
        # q predicts gravity in the body frame.  Initializing from the static
        # acceleration vector gives an immediate neutral result at calibration.
        self.q = _q_from_two_vectors(np.array([0.0, 0.0, 1.0]), initial_accel)

    def update(self, gyro_rad_s: np.ndarray, accel: np.ndarray, dt: float,
               use_accel: bool) -> np.ndarray:
        q1, q2, q3, q4 = self.q
        gx, gy, gz = gyro_rad_s
        q_dot = 0.5 * np.array([
            -q2 * gx - q3 * gy - q4 * gz,
            q1 * gx + q3 * gz - q4 * gy,
            q1 * gy - q2 * gz + q4 * gx,
            q1 * gz + q2 * gy - q3 * gx,
        ])

        if use_accel:
            ax, ay, az = _normalize(accel)
            if np.linalg.norm([ax, ay, az]) > 0.0:
                f = np.array([
                    2.0 * (q2 * q4 - q1 * q3) - ax,
                    2.0 * (q1 * q2 + q3 * q4) - ay,
                    2.0 * (0.5 - q2 * q2 - q3 * q3) - az,
                ])
                jacobian = np.array([
                    [-2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2],
                    [2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3],
                    [0.0, -4.0 * q2, -4.0 * q3, 0.0],
                ])
                step = jacobian.T @ f
                step_norm = float(np.linalg.norm(step))
                if step_norm > 1e-12:
                    q_dot -= self.beta * step / step_norm

        self.q = _q_normalize(self.q + q_dot * dt)
        return self.q.copy()
