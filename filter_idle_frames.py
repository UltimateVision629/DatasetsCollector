"""
Filter out idle/near-zero-action frames from training data.

A frame is kept if EITHER arm has meaningful movement:
  - Position delta magnitude > pos_threshold
  - Rotation delta magnitude > rot_threshold
  - Gripper change vs previous frame > grip_threshold

Outputs new .npz files with only active frames. Original data is NOT modified.

Usage:
  cd DatasetsCollector

  # 推荐阈值（过滤 ~30% 发呆帧）
  python filter_idle_frames.py --input_dir ./demos --output_dir ./demos_filtered --pos_threshold 0.0015 --rot_threshold 0.005

  # 更激进（过滤 ~50%）
  python filter_idle_frames.py --input_dir ./demos --output_dir ./demos_filtered --pos_threshold 0.002 --rot_threshold 0.01

  # 从 network 项目调用
  python ../DatasetsCollector/filter_idle_frames.py --input_dir ../datasets/trajectories --output_dir ../datasets/trajectories_filtered

Threshold tuning:
  - 位移阈值 (--pos_threshold): 单帧 EEF 位移均值 ~0.001m，推荐 0.001-0.002
  - 旋转阈值 (--rot_threshold): 单帧 EEF 旋转均值 ~0.003rad，推荐 0.003-0.01
  - 夹爪阈值 (--grip_threshold): 0=开 1=关，推荐 0.01（检测开关切换）

Output:
  过滤后的 .npz 文件存到 --output_dir，文件名不变，帧索引重新从 0 编号。
  终端打印每条轨迹的裁切统计和汇总。
"""

import argparse
import glob
import os
import sys

import numpy as np


def is_active_frame(action_14d: np.ndarray, prev_action_14d: np.ndarray = None,
                    pos_threshold: float = 0.0015, rot_threshold: float = 0.005,
                    grip_threshold: float = 0.01) -> bool:
    """
    Check if a 14-dim action frame has meaningful movement.

    action_14d: [R_dx,R_dy,R_dz,R_dRx,R_dRy,R_dRz,R_grip, L_dx,L_dy,L_dz,L_dRx,L_dRy,L_dRz,L_grip]
    Returns True if any arm moves meaningfully.
    """
    for arm_start in [0, 7]:  # 0=right arm, 7=left arm
        dx = action_14d[arm_start]
        dy = action_14d[arm_start + 1]
        dz = action_14d[arm_start + 2]
        dRx = action_14d[arm_start + 3]
        dRy = action_14d[arm_start + 4]
        dRz = action_14d[arm_start + 5]

        pos_mag = np.sqrt(dx * dx + dy * dy + dz * dz)
        rot_mag = np.sqrt(dRx * dRx + dRy * dRy + dRz * dRz)

        if pos_mag > pos_threshold or rot_mag > rot_threshold:
            return True

    # Check gripper change vs previous frame
    if prev_action_14d is not None:
        for gripper_dim in [6, 13]:  # right gripper, left gripper
            if abs(action_14d[gripper_dim] - prev_action_14d[gripper_dim]) > grip_threshold:
                return True

    return False


def filter_trajectory(npz_path: str, output_dir: str,
                      pos_threshold: float, rot_threshold: float,
                      grip_threshold: float) -> dict:
    """Filter one trajectory, save filtered version. Returns stats."""
    data = np.load(npz_path, allow_pickle=True)
    T = int(data["length"].item()) if "length" in data else 50
    name = os.path.splitext(os.path.basename(npz_path))[0]

    # Find active frames
    active_indices = []
    prev_action = None
    for t in range(T):
        action = data[f"action/{t}"].astype(np.float32)
        if is_active_frame(action, prev_action, pos_threshold, rot_threshold, grip_threshold):
            active_indices.append(t)
        prev_action = action

    n_kept = len(active_indices)
    n_total = T

    if n_kept == 0:
        print(f"  {name}: {n_total} frames -> 0 kept (SKIPPED - all idle)")
        return {"name": name, "total": n_total, "kept": 0, "dropped": n_total}

    # Collect observation key prefixes
    obs_keys = set()
    for key in data.keys():
        if key.startswith("obs/"):
            obs_keys.add("/".join(key.split("/")[:2]))

    # Build filtered data
    filtered = {}

    # Per-frame data: stack and re-index
    for ok in sorted(obs_keys):
        frames = [data[f"{ok}/{t}"] for t in active_indices]
        stacked = np.stack(frames, axis=0)
        for new_t in range(n_kept):
            filtered[f"{ok}/{new_t}"] = stacked[new_t]

    # Actions
    action_frames = [data[f"action/{t}"] for t in active_indices]
    stacked = np.stack(action_frames, axis=0)
    for new_t in range(n_kept):
        filtered[f"action/{new_t}"] = stacked[new_t]

    # Non-frame data: copy as-is
    for key in data.keys():
        if key.startswith("obs/") or key.startswith("action/") or key == "length":
            continue
        val = data[key]
        filtered[key] = val.item() if isinstance(val, np.ndarray) and val.ndim == 0 else val

    filtered["length"] = np.array(n_kept)

    # Save
    out_path = os.path.join(output_dir, f"{name}.npz")
    np.savez_compressed(out_path, **filtered)

    drop_pct = (n_total - n_kept) / n_total * 100
    print(f"  {name}: {n_total} -> {n_kept} frames ({drop_pct:.0f}% dropped)")

    return {"name": name, "total": n_total, "kept": n_kept, "dropped": n_total - n_kept}


def main():
    parser = argparse.ArgumentParser(
        description="Filter idle frames from training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_idle_frames.py -i ./demos -o ./demos_filtered
  python filter_idle_frames.py -i ./demos -o ./demos_filtered --pos_threshold 0.002 --rot_threshold 0.01
        """,
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory containing .npz trajectory files")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save filtered .npz files (original NOT modified)")
    parser.add_argument("--pos_threshold", type=float, default=0.0015,
                        help="Position delta magnitude threshold in meters (default: 0.0015)")
    parser.add_argument("--rot_threshold", type=float, default=0.005,
                        help="Rotation delta magnitude threshold in radians (default: 0.005)")
    parser.add_argument("--grip_threshold", type=float, default=0.01,
                        help="Gripper change threshold [0-1] (default: 0.01)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if not files:
        print(f"ERROR: No .npz files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input:  {args.input_dir} ({len(files)} trajectories)")
    print(f"Output: {args.output_dir}")
    print(f"Thresholds: pos>{args.pos_threshold}m, rot>{args.rot_threshold}rad, grip_change>{args.grip_threshold}")
    print()

    all_stats = []
    total_kept, total_dropped = 0, 0

    for f in files:
        stats = filter_trajectory(f, args.output_dir,
                                  args.pos_threshold, args.rot_threshold,
                                  args.grip_threshold)
        all_stats.append(stats)
        total_kept += stats["kept"]
        total_dropped += stats["dropped"]

    total = total_kept + total_dropped
    kept_pct = total_kept / max(total, 1) * 100
    dropped_pct = total_dropped / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"Summary: {total} -> {total_kept} kept ({kept_pct:.1f}%),  {total_dropped} dropped ({dropped_pct:.1f}%)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
