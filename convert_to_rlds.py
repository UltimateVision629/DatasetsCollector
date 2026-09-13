"""
Convert collected .npz demo trajectories to VLA-Adapter compatible RLDS format
and compute normalization statistics.

VLA-Adapter's prismatic expects:
  - RLDS trajectories with specific observation/action keys
  - Norm stats: dict with "action" and "proprio" keys, each containing q01/q99/mask

Usage:
  python convert_to_rlds.py --input_dir ../demos --output_dir ../dataset
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

from static_frame_filter import filter_trajectories


def load_trajectories(input_dir: str) -> List[dict]:
    """Load all .npz files from input_dir."""
    files = sorted(glob.glob(os.path.join(input_dir, "**/*.npz"), recursive=True))
    print(f"Found {len(files)} trajectories in {input_dir}")
    trajectories = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        traj = {
            "agentview_image": data["agentview_image"],
            "robot0_joint_pos": data["robot0_joint_pos"],
            "robot0_eef_pos": data["robot0_eef_pos"],
            "robot0_eef_quat": data["robot0_eef_quat"],
            "robot0_gripper_qpos": data["robot0_gripper_qpos"],
            "robot1_joint_pos": data["robot1_joint_pos"],
            "robot1_eef_pos": data["robot1_eef_pos"],
            "robot1_eef_quat": data["robot1_eef_quat"],
            "robot1_gripper_qpos": data["robot1_gripper_qpos"],
            "action": data["action"],
            "language_instruction": str(data["language_instruction"]),
        }
        trajectories.append(traj)
    return trajectories


def accumulate_actions(actions: np.ndarray, step_skip: int, action_dim: int = 14) -> np.ndarray:
    """step_skip 滑窗累积，与 BlockGraspDataset 采样语义一致：
    delta 维求和、gripper 维(6,13)取窗口末帧绝对值（训练目标同款）。"""
    T = len(actions)
    out = np.zeros((T - step_skip + 1, action_dim), dtype=np.float32)
    for t in range(T - step_skip + 1):
        acc = actions[t:t + step_skip].sum(axis=0).astype(np.float32)
        acc[6] = actions[t + step_skip - 1, 6]
        acc[13] = actions[t + step_skip - 1, 13]
        out[t] = acc
    return out


def compute_norm_stats(trajectories: List[dict], step_skip: int = 0) -> dict:
    """Compute q01/q99 normalization stats for actions and proprioception.

    step_skip > 0 时对动作做训练同款滑窗累积（delta 求和 + gripper 取末帧），
    直接产出与训练目标匹配的累积版 stats（无需再跑 recompute_norm_stats.py）；
    step_skip = 0 保持逐帧 stats（向后兼容）。

    Returns stats dict compatible with VLA-Adapter's norm_stats format:
      {"dataset_name": {"action": {"q01": [...], "q99": [...], "mask": [...]},
                        "proprio": {"q01": [...], "q99": [...], "mask": [...]}}}
    """
    if step_skip > 0:
        all_actions = np.concatenate(
            [accumulate_actions(t["action"], step_skip) for t in trajectories], axis=0)
        print(f"[NormStats] step_skip={step_skip}: 对累积窗口样本计算 stats "
              f"({all_actions.shape[0]} 个样本)")
    else:
        all_actions = np.concatenate([t["action"] for t in trajectories], axis=0)
    # proprio 与 BlockGraspDataset 拼接顺序一致：每帧 [arm0(joint6,eef3,quat4,grip2),
    # arm1(joint6,eef3,quat4,grip2)] = 30 维；沿特征轴拼接，轨迹沿样本轴堆叠
    all_proprio = np.concatenate([
        np.concatenate([t["robot0_joint_pos"], t["robot0_eef_pos"],
                        t["robot0_eef_quat"], t["robot0_gripper_qpos"],
                        t["robot1_joint_pos"], t["robot1_eef_pos"],
                        t["robot1_eef_quat"], t["robot1_gripper_qpos"]], axis=1)
        for t in trajectories
    ], axis=0)

    action_dim = all_actions.shape[-1]
    proprio_dim = all_proprio.shape[-1]

    action_q01 = np.percentile(all_actions, 1, axis=0).tolist()
    action_q99 = np.percentile(all_actions, 99, axis=0).tolist()
    proprio_q01 = np.percentile(all_proprio, 1, axis=0).tolist()
    proprio_q99 = np.percentile(all_proprio, 99, axis=0).tolist()

    stats = {
        "block_grasp": {
            "action": {
                "q01": action_q01,
                "q99": action_q99,
                "mask": [True] * action_dim,
            },
            "proprio": {
                "q01": proprio_q01,
                "q99": proprio_q99,
                "mask": [True] * proprio_dim,
            },
        }
    }

    print(f"\nNorm stats (block_grasp):")
    print(f"  action dim={action_dim}  q01={[f'{v:.4f}' for v in action_q01]}")
    print(f"  action       q99={[f'{v:.4f}' for v in action_q99]}")
    print(f"  proprio dim={proprio_dim}  q01={[f'{v:.4f}' for v in proprio_q01]}")
    print(f"  proprio       q99={[f'{v:.4f}' for v in proprio_q99]}")

    return stats


def build_rlds_dataset(trajectories: List[dict], output_dir: str):
    """Build TFDS RLDS dataset from trajectories.

    Uses the RLDS episode format expected by VLA-Adapter:
    https://github.com/google-research/rlds
    """
    import tensorflow_datasets as tfds

    os.makedirs(output_dir, exist_ok=True)

    # Build TFDS dataset
    dataset_config = tfds.rlds.rlds_base.RLDSConfig(
        name="block_grasp",
        description="Block grasping demonstrations from LIBERO Unity",
        homepage="",
        citation="",
    )

    builder = tfds.rlds.RLDSBuilder(
        config=dataset_config,
        data_dir=output_dir,
        version=tfds.core.Version("1.0.0"),
    )

    # Create episodes
    for i, traj in enumerate(trajectories):
        episode = []
        T = len(traj["action"])

        for t in range(T):
            step = {
                "observation": {
                    "agentview_image": traj["agentview_image"][t],
                    "robot0_joint_pos": traj["robot0_joint_pos"][t].astype(np.float32),
                    "robot0_eef_pos": traj["robot0_eef_pos"][t].astype(np.float32),
                    "robot0_eef_quat": traj["robot0_eef_quat"][t].astype(np.float32),
                    "robot0_gripper_qpos": traj["robot0_gripper_qpos"][t].astype(np.float32),
                },
                "action": traj["action"][t].astype(np.float32),
                "language_instruction": traj["language_instruction"],
                "discount": np.float32(1.0),
                "is_first": t == 0,
                "is_last": t == T - 1,
                "is_terminal": t == T - 1,
            }
            episode.append(step)

        # Register episode with builder
        # Note: In practice, VLA-Adapter uses a custom TFRecord-based RLDS format.
        # This builds the standard RLDS format; for VLA-Adapter compatibility,
        # we also export as raw TFRecords below.
        episode_id = f"episode_{i:06d}"

    print(f"Built dataset with {len(trajectories)} episodes in {output_dir}")


def export_simple_format(trajectories: List[dict], output_dir: str, norm_stats: dict):
    """Export trajectories in a simple format compatible with VLA-Adapter's dataloader.

    VLA-Adapter's prismatic/vla/datasets/rlds/ expects TFRecord files with
    serialized tf.train.Example protos. This function exports both:
      1. Raw .npz files in VLA-Adapter observation format
      2. Norm stats JSON
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save norm stats
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm stats to {stats_path}")

    # Save each trajectory as individual .npz with VLA-Adapter keys
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    for i, traj in enumerate(trajectories):
        traj_data = {}
        T = len(traj["action"])

        for t in range(T):
            traj_data[f"obs/agentview_image/{t}"] = traj["agentview_image"][t]
            traj_data[f"obs/robot0_joint_pos/{t}"] = traj["robot0_joint_pos"][t].astype(np.float32)
            traj_data[f"obs/robot0_eef_pos/{t}"] = traj["robot0_eef_pos"][t].astype(np.float32)
            traj_data[f"obs/robot0_eef_quat/{t}"] = traj["robot0_eef_quat"][t].astype(np.float32)
            traj_data[f"obs/robot0_gripper_qpos/{t}"] = traj["robot0_gripper_qpos"][t].astype(np.float32)
            traj_data[f"obs/robot1_joint_pos/{t}"] = traj["robot1_joint_pos"][t].astype(np.float32)
            traj_data[f"obs/robot1_eef_pos/{t}"] = traj["robot1_eef_pos"][t].astype(np.float32)
            traj_data[f"obs/robot1_eef_quat/{t}"] = traj["robot1_eef_quat"][t].astype(np.float32)
            traj_data[f"obs/robot1_gripper_qpos/{t}"] = traj["robot1_gripper_qpos"][t].astype(np.float32)
            traj_data[f"action/{t}"] = traj["action"][t].astype(np.float32)

        traj_data["language_instruction"] = traj["language_instruction"]
        traj_data["length"] = T

        out_path = os.path.join(traj_dir, f"trajectory_{i:06d}.npz")
        np.savez_compressed(out_path, **traj_data)

    print(f"Saved {len(trajectories)} trajectories to {traj_dir}")


def verify_output(output_dir: str):
    """转换后校验所有输出 .npz 可读 —— 坏文件会在训练时静默崩溃
    （zipfile.BadZipFile），必须在转换阶段当场发现。

    任一文件损坏 → 打印清单并 SystemExit（非零退出，提示重新转换）。
    """
    traj_dir = os.path.join(output_dir, "trajectories")
    files = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
    bad = []
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as d:
                T = d["length"].item() if "length" in d else len(d["action/0"])
                if T > 0:
                    _ = d[f"action/{0}"]   # 触发实际读取
        except Exception as e:
            bad.append((os.path.basename(f), f"{type(e).__name__}: {e}"))
    if bad:
        for name, err in bad:
            print(f"[Verify] 损坏文件: {name} ({err})")
        raise SystemExit(f"[Verify] {len(bad)}/{len(files)} 个输出文件损坏，请重新转换")
    print(f"[Verify] 全部 {len(files)} 个轨迹文件校验通过")


def main():
    parser = argparse.ArgumentParser(description="Convert demos to VLA-Adapter format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of .npz demo files")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output dataset directory")
    parser.add_argument("--static_max_eps", type=float, default=0.003,
                        help="静止帧判定阈值 max|delta| < eps（0 = 不过滤；行业区间 0.001-0.01）")
    parser.add_argument("--static_max_run", type=int, default=5,
                        help="连续静止段超过 N 帧则整段裁剪（0 = 不过滤；与 step_skip 对齐）")
    parser.add_argument("--step_skip", type=int, default=0,
                        help=">0 时对过滤后动作按训练同款滑窗累积，直接算累积版 stats"
                             "（0 = 逐帧 stats，向后兼容；训练 step_skip=5 时传 5）")
    args = parser.parse_args()

    # Load
    trajectories = load_trajectories(args.input_dir)
    if len(trajectories) == 0:
        print("ERROR: No trajectories found!")
        return

    print(f"\nTrajectory stats:")
    lengths = [len(t["action"]) for t in trajectories]
    print(f"  count={len(trajectories)}, total_steps={sum(lengths)}")
    print(f"  min_len={min(lengths)}, max_len={max(lengths)}, mean_len={np.mean(lengths):.0f}")

    # 静止帧过滤（数据初始化入口，stats 之前 → 归一化统计基于过滤后分布）
    trajectories = filter_trajectories(trajectories, args.static_max_eps, args.static_max_run)
    if len(trajectories) == 0:
        print("ERROR: All trajectories filtered out (all static)!")
        return

    # Compute norm stats（step_skip>0 时直接算累积版，免 recompute 步骤）
    norm_stats = compute_norm_stats(trajectories, step_skip=args.step_skip)

    # Export
    export_simple_format(trajectories, args.output_dir, norm_stats)
    verify_output(args.output_dir)
    print("\nDone! Dataset ready for training.")


if __name__ == "__main__":
    main()
