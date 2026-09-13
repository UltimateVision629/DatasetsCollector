"""回归测试：lerobot 式节拍工具（precise_sleep / obs_fingerprint / LoopTiming）。

不需要 Unity / Joy-Con / conda 环境即可运行：
    python DatasetsCollector/test_collect_timing.py

覆盖：
  1. collect_datasets.py 整文件语法
  2. 工具块可独立执行（只依赖 time/platform/numpy）
  3. precise_sleep 在 60Hz(16.7ms) 节拍上的精度
  4. obs_fingerprint 能区分不同观测（重复帧检测的前提）
  5. LoopTiming 的超时占比 / obs 重复率统计与清零语义
"""
import ast
import os
import platform
import time

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_datasets.py")


def load_utils():
    src = open(SRC, encoding="utf-8").read()
    ast.parse(src)
    print("[1] collect_datasets.py 语法 OK")
    start = src.index("# ── lerobot 式节拍工具")
    end = src.index("# ── Quaternion math")
    ns = {"time": time, "platform": platform, "np": np, "COLLECT_FPS": 60.0}
    exec(compile(src[start:end], "<timing-block>", "exec"), ns)
    print("[2] 工具块可独立执行 OK")
    return ns["precise_sleep"], ns["LoopTiming"], ns["obs_fingerprint"]


def main():
    precise_sleep, LoopTiming, obs_fingerprint = load_utils()

    for target in (0.0167, 0.010, 0.005):
        for _ in range(5):                       # 预热，避免冷启动调度噪声
            precise_sleep(target)
        errs = []
        for _ in range(20):
            t0 = time.perf_counter()
            precise_sleep(target)
            errs.append(time.perf_counter() - t0 - target)
        print(f"[3] precise_sleep({target * 1000:.1f}ms): 误差均值 {np.mean(errs) * 1000:+.2f}ms "
              f"最大 {max(errs) * 1000:+.2f}ms")
        assert abs(np.mean(errs)) < 0.002, f"precise_sleep 偏差过大: {np.mean(errs)}"

    t0 = time.perf_counter()
    precise_sleep(0.0)
    assert time.perf_counter() - t0 < 0.001

    oa = {"robot0_joint_pos": np.arange(6, dtype=float),
          "robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    ob = {"robot0_joint_pos": np.arange(6, dtype=float) + 1,
          "robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    assert obs_fingerprint(oa) != obs_fingerprint(ob)
    print("[4] obs_fingerprint 区分不同观测 OK")

    lt = LoopTiming(60.0, report_every_s=999, warn_every_s=999)
    for i in range(60):
        lt.note_tick(0.030)                      # 33Hz 实际 vs 60Hz 目标
        lt.note_obs(oa if (i // 2) % 2 == 0 else ob)   # 每 2 帧换 → 50% 重复
    s = lt.episode_summary()
    print("[5] episode_summary:", s)
    assert s["loop_fps_target"] == 60.0
    assert s["loop_overrun_pct"] == 100.0
    assert 45.0 <= s["loop_obs_dup_pct"] <= 55.0
    assert s["loop_ticks"] == 60

    lt2 = LoopTiming(60.0, report_every_s=999, warn_every_s=999)
    for _ in range(10):
        lt2.note_tick(0.010)
        lt2.note_obs(oa)
    first, second = lt2.episode_summary()["loop_ticks"], lt2.episode_summary()["loop_ticks"]
    assert first == 10 and second == 0, (first, second)
    print(f"[6] summary 取完即清零 OK (首次 {first} → 再次 {second})")

    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
