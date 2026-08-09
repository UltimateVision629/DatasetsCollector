"""静止帧过滤（行业标准 max 判据，转换时一次性执行）。

判据（方案1，基于动作信号）：
  per-frame `max over 12 个 delta 维 |a_d| < eps` 判定为静止帧。
  eps 默认 0.002（行业区间 0.001-0.01；本数据集实测 0.001 过紧、0.005 过松）。

裁剪规则（行业做法，见 DROID / Open X-Embodiment / LIBERO-no_noops / LeRobot）：
  1. 起始连续静止段全裁 —— 杀"home 图像 → 0 动作"过学习（模型无法按任务发起的根因）
  2. 连续静止段 > max_run 帧全裁 —— 中程/收尾长空闲段（避免模型学到"不动是最优"）
  3. 短停顿（≤max_run 帧）保留 —— 合法"保持/对准"教学 + 动作噪声 dip 不误伤

纯 numpy，无依赖；train.py 零改动，过滤结果固化在 convert 产物里。
"""
from typing import Dict, List, Optional

import numpy as np

# 14 维动作中的 delta 维（排除 gripper 6/13 —— 0/1 二值目标不参与静止判定）
DELTA_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def compute_static_mask(actions: np.ndarray, eps: float = 0.002) -> np.ndarray:
    """actions: [T, 14] 逐帧动作 → [T] bool（True = 静止）。

    max(|a_d|) < eps：任何一维在动即视为非静止（不被旋转维主导、不被单维掩盖）。
    """
    return np.abs(actions[:, DELTA_DIMS]).max(axis=1) < eps


def trim_static_frames(static: np.ndarray, max_run: int = 16) -> np.ndarray:
    """[T] static 掩码 → [T] keep 掩码（True = 保留该帧）。

    规则：起始静止段全裁；其余连续静止段 > max_run 帧全裁；短停顿保留。
    默认全保留，仅按规则置 False（不能从 ~static 出发 —— 那会把所有
    静止帧默认排除，短停顿就留不住了）。
    """
    keep = np.ones(len(static), dtype=bool)
    T = len(static)
    i = 0
    while i < T:
        if static[i]:
            j = i
            while j < T and static[j]:
                j += 1
            run_len = j - i
            if i == 0 or run_len > max_run:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return keep


def filter_trajectory(traj: dict, eps: float = 0.002, max_run: int = 16) -> Optional[dict]:
    """按 keep 掩码裁剪 traj 的全部数组键并重排下标。

    eps <= 0 或 max_run <= 0 → 不过滤，原样返回。
    全静止（keep 为空）→ 返回 None（整条丢弃，行业做法：剔除全程无运动的无效轨迹）。
    """
    if eps <= 0 or max_run <= 0:
        return traj
    actions = traj["action"]
    static = compute_static_mask(actions, eps)
    keep = trim_static_frames(static, max_run)
    if not keep.any():
        return None
    filtered = {}
    for k, v in traj.items():
        if k == "language_instruction":
            filtered[k] = v
        elif isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == len(actions):
            filtered[k] = v[keep]
        else:
            filtered[k] = v
    return filtered


def filter_trajectories(trajectories: List[dict], eps: float = 0.002,
                        max_run: int = 16) -> List[dict]:
    """批量过滤。打印每条裁剪前后长度与整体统计，返回过滤后的轨迹列表。"""
    if eps <= 0 or max_run <= 0:
        print(f"[StaticFilter] 已禁用（eps={eps}, max_run={max_run}），{len(trajectories)} 条原样保留")
        return trajectories

    out = []
    before_steps = after_steps = 0
    for traj in trajectories:
        T0 = len(traj["action"])
        before_steps += T0
        ft = filter_trajectory(traj, eps, max_run)
        if ft is None:
            print(f"[StaticFilter] 丢弃全静止轨迹: {T0} 帧 ({traj.get('language_instruction', '?')})")
            continue
        T1 = len(ft["action"])
        after_steps += T1
        if T1 != T0:
            print(f"[StaticFilter] {traj.get('language_instruction', '?'):42s} "
                  f"{T0:4d} → {T1:4d} 帧（裁 {T0 - T1:4d}）")
        out.append(ft)

    print(f"[StaticFilter] eps={eps}, max_run={max_run}: "
          f"{len(trajectories)} 条 → {len(out)} 条, "
          f"{before_steps} → {after_steps} 帧（保留 {100 * after_steps / max(before_steps, 1):.1f}%）")
    return out


if __name__ == "__main__":
    # 自测：构造样例验证三条规则（静止段被运动帧分隔才独立成段）
    #   0-2  静止（起始段 → 裁）
    #   3-9  运动
    #   10-11 静止 2 帧（短停顿，两侧运动 → 保留）
    #   12-13 运动
    #   14-33 静止 20 帧（>16 → 裁）
    #   34-35 运动
    acts = np.zeros((36, 14))
    for rng in [(3, 10), (12, 14), (34, 36)]:
        acts[rng[0]:rng[1], 0] = 0.005
    static = compute_static_mask(acts, 0.002)
    keep = trim_static_frames(static, 16)
    print("static:", "".join("S" if s else "." for s in static))
    print("keep  :", "".join("K" if k else "-" for k in keep))
    assert keep[:3].sum() == 0, "起始静止段应全裁"
    assert keep[3:10].all() and keep[12:14].all() and keep[34:].all(), "运动段应保留"
    assert keep[10:12].all(), "短停顿(2帧)应保留"
    assert keep[14:34].sum() == 0, "长静止段(20帧)应全裁"
    print("自测 OK")
