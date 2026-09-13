"""验证采集数据里 label 累计位移 vs MuJoCo 实际位移的一致性。

背景：采集时如果用了 lerobot IK 后端，操作员的视觉补偿会被固化进
label（推杆意图 ≠ MuJoCo 实际位移），推理时用 mujoco 后端执行 label
会产生系统性过冲/欠冲。位移范数在坐标系旋转下不变，可直接比较。

判读：
- 比值都在 0.9-1.1    → 补偿不大，lerobot 采集没问题
- 比值明显 >1.2 或 <0.8 → 补偿被固化进 label，应改用 mujoco 后端采集

用法：
  python check_label_vs_eef.py [--dir demos] [--limit 12]
"""

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="demos", help="demos 目录（默认 demos）")
    ap.add_argument("--limit", type=int, default=20, help="只检查前 N 条（默认 20）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))[:args.limit]
    if not files:
        print(f"错误: {args.dir} 下没有 .npz")
        return

    ratios = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            action = np.asarray(d["action"], dtype=np.float64)
            eef = np.asarray(d["robot0_eef_pos"], dtype=np.float64)

        lab = action[:, :3].sum(axis=0)      # label 累计位移（右臂，world 系）
        act = eef[-1] - eef[0]               # MuJoCo 实际位移
        norm_act = np.linalg.norm(act)
        norm_lab = np.linalg.norm(lab)

        if norm_act < 1e-3:
            print(f"{os.path.basename(f)}: eef 无效（旧数据/未动），跳过")
            continue
        r = norm_lab / norm_act
        ratios.append(r)
        flag = "✅" if 0.9 <= r <= 1.1 else "⚠️"
        print(f"{os.path.basename(f)}: label {norm_lab:.3f} m vs 实际 {norm_act:.3f} m → 比值 {r:.2f} {flag}")

    if ratios:
        med = float(np.median(ratios))
        print(f"\n中位比值: {med:.2f}  （{'✅ 0.9-1.1，lerobot 采集可用' if 0.9 <= med <= 1.1 else '⚠️ 偏差大，建议 mujoco 后端采集'}）")


if __name__ == "__main__":
    main()