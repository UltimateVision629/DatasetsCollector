# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the data collection side of a dual-arm robotic block-grasping system. It uses Joy-Con teleop to control SO100 robots in Unity via TCP, records (observation, action) trajectories as `.npz` files, and provides format conversion for training.

The Unity simulator lives at `../libero-unity` and the ML training side at `../network`.

## Project structure

```
DatasetsCollector\
    collect_datasets.py     # Joy-Con teleop data collection
    env_client.py           # TCP client → Unity TrainingServer (5556)
    convert_to_rlds.py      # .npz → training format + norm stats
    npz_visualizer.py       # Data visualization GUI (RoboView)
    demos\                  # Collected .npz output
```

## Architecture: LeRobot-style processor separation

The collection script (`collect_datasets.py`) splits the Joy-Con output into two independent pipelines:

```
Joy-Con get_control() → target_pose, gripper, button
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
 teleop_action_processor          robot_action_processor
 (_compute_action_label)          (_compute_robot_command)
 target_pose → EEF delta          target_pose → IK → joints
            │                               │
            ▼                               ▼
      action_label                    send_joints() → Unity
      (saved to .npz)                (JoyConReceiver:5555)
```

**Key design**: Action labels are **SENDER-side** (Joy-Con target deltas), matching LeRobot/LIBERO philosophy. Labels represent operator intent. The robot command (joint angles) is a separate path — they can differ.

## TCP protocol

Two separate TCP servers, do NOT confuse them:

| Server | Port | Direction | Protocol |
|--------|------|-----------|----------|
| JoyConReceiver | 5555 | Python → Unity | JSON: `{"robot_0":{"joints":[...],"gripper":x,"button":0}}` |
| TrainingServer | 5556 | bidirectional | JSON-line: `{"cmd":"reset"/"step"/"get_obs"}` |

TrainingServer commands (`env_client.py`):
- `{"cmd":"reset"}` — reset environment, return observation JSON
- `{"cmd":"step","action":[...]}` — execute action, return `{obs, reward, done, success, step}`
- `{"cmd":"get_obs"}` — return current observation without stepping
- `{"cmd":"get_task"}` — return `{"language_instruction":"..."}`

Observations include: `agentview_image` (base64 PNG, 224×224 letterboxed), robot0/robot1 `joint_pos`, `eef_pos`, `eef_quat`, `gripper_qpos`.

## Action space

**Format**: Dual-arm EEF delta — 14 dimensions (7 per arm × 2 arms).
Per arm: `[dx, dy, dz, dRx, dRy, dRz, gripper]`

- `dx, dy, dz`: EEF position delta in meters
- `dRx, dRy, dRz`: axis-angle rotation delta = rotation axis × rotation angle in radians
- `gripper`: absolute gripper target — **joyconrobotics 语义: 1=open, 0=close**（`joyconrobotics.py:198` 源码注释明确；旧文档写成 0=open 是错的）

**Action order in 14-dim array**: `[R_dx,R_dy,R_dz,R_dRx,R_dRy,R_dRz,R_grip, L_dx,L_dy,L_dz,L_dRx,L_dRy,L_dRz,L_grip]`

### Gripper 执行语义（勿与 label 混淆）

- 训练 label 是 sender-side 意图（上面 0/1）
- **发送给 Unity 的 Jaw 目标 = raw 0/1 直通**（`target_qpos` 最后一个元素）——Unity `MjJoyConController.SetJoint("Jaw", v)` 把值原样写 MuJoCo position actuator ctrl
- **MuJoCo Jaw 关节/ctrl 范围**: `[-0.174, 1.75]` rad，-0.174 = 闭合位、1.75 = 全开位。raw 直通意味着 0 → 行程中间（半开）、1 → 近全开——**物理上爪到不了 -0.174 完全闭合位**，这是已知的旧行为，数据 label 不受影响
- 对比：`replay_action_via_ik.py` 有 `gripper_to_jaw()`（`-0.174 + 1.924*g`）让重放时爪真正闭合；**collect 刻意保持 raw 直通**（2026-08-06 用户决定不重采，勿改回映射，除非重新采集）

## Joy-Con teleop

Joy-Con `get_control()` returns `target_pose = [x_r, _, z_r, roll_r, pitch_r, yaw_r]` plus `gripper_state`.

### Teleop action processor: `_compute_action_label(arm_index, target_pose, gripper)`

Computes sender-side 7-dim EEF delta from consecutive Joy-Con target poses:
1. Applies orientation transforms: `pitch_r = -pitch_r`, `roll_r = roll_r - π/2`
2. Converts local cylindrical position to base-relative Cartesian: `[x_r, 0.01, z_r]`
3. Rotates by base yaw to world-frame
4. Wrist orientation via `_euler_zxy_to_quat()` (matches Unity `Quaternion.Euler` extrinsic Z-X-Y)
5. Full world orientation = `quat_multiply(q_base_yaw, q_wrist)`
6. First frame: stores initial pose, returns zero delta
7. Subsequent frames: position delta by subtraction, rotation delta via quaternion diff → axis-angle

### Robot action processor: `_compute_robot_command(arm_index, target_pose, gripper)`

Delegates to the **pluggable IK backend** (`libero-unity/test/arm_ik.py`, `--ik-backend {lerobot,mujoco,placo}`, 默认 `lerobot` 保持原行为；`mujoco` 为 MuJoCo 精确关节角，与 Unity 同一模型):
```python
joints5, new_q = self.ik[arm_index].solve(target_pose, gripper_state, current_arm_q)
target_qpos = np.concatenate((joints5, [gripper_state]))  # raw gripper 直通
```

每个臂一个 IK 实例（MuJoCoIK 内部跟踪 per-arm lerobot 翻译 warm，episode 重置时调用 `reset_warm()`）。

**IMPORTANT**: `[x_r, y_r, z_r]` is in a LOCAL cylindrical coordinate frame that rotates with the base joint (yaw). It is NOT the same as the MuJoCo world-frame `obs["robot0_eef_pos"]`.

### Helper: `_euler_zxy_to_quat(rx, ry, rz)`

Converts extrinsic Z-X-Y Euler angles to quaternion [x, y, z, w], matching Unity's `Quaternion.Euler(rx, ry, rz)` convention. Quaternion formula: `q = q_y ⊗ q_x ⊗ q_z`.

### Home reset (`_go_home`)

On A/Y button press, resets IK warm-start to `INIT_ARM_Q` (5-DOF, from `arm_ik.py`), computes home joint angles via IK, sends via JoyConReceiver, and resets Joy-Con internal state:
- `jc.set_position(list(_SO100_HOME_XYZ))` — reset position accumulator
- `jc.yaw_diff = 0.0` — reset accumulated yaw
- `jc.gripper_state = jc.gripper_open` — reset gripper to open

### Button controls

- **A button** (right Joy-Con) → save episode as success + go home
- **Y button** (right Joy-Con) → discard episode + go home
- **ZL / ZR** → toggle gripper open/close

## Main loop (60 Hz)

```
1. Read Joy-Cons once per frame (get_control())
2. _compute_action_label() → EEF delta labels
3. _compute_robot_command() → IK joint angles
4. Send joint angles via JoyConReceiver (5555)
5. env.get_obs() from TrainingServer (5556)
6. Record (obs, action_label) — label = sender-side EEF delta
7. Button events: A=save, Y=discard
```

## Data format (.npz files)

Collected demos (`demos/`):
- `agentview_image`: [T, 224, 224, 3] uint8
- `robot0_joint_pos`, `robot1_joint_pos`: [T, 6] float32
- `robot0_eef_pos`, `robot1_eef_pos`: [T, 3] float32
- `robot0_eef_quat`, `robot1_eef_quat`: [T, 4] float32 (Unity order: [x, y, z, w])
- `robot0_gripper_qpos`, `robot1_gripper_qpos`: [T, 2] float32
- `action`: [T, 14] float32
- `language_instruction`: str
- `success`: bool

## Format conversion (convert_to_rlds.py)

Converts `.npz` demos to VLA-Adapter training format and computes normalization stats (q01/q99):

```bash
python convert_to_rlds.py --input_dir ./demos --output_dir ../network/dataset
```

Output:
- `dataset/trajectories/trajectory_XXXXXX.npz` — per-timestep keys (`obs/agentview_image/{t}`, `action/{t}`, etc.)
- `dataset/dataset_statistics.json` — norm stats for training

## RoboView visualizer (npz_visualizer.py)

The visualizer has two code paths sharing the same data-loading and plotting utilities:

### Entry points

| Path | Entry | Window |
|------|-------|--------|
| `python npz_visualizer.py` (no args) | `main()` → `launch_import_app()` → `RoboViewApp` | tkinter import window → embedded-matplotlib analysis window |
| `python npz_visualizer.py demos --episode-index 1` | `main()` → `visualize(ep)` | pure matplotlib `plt.show()` window |
| `python npz_visualizer.py <file.npz>` | `main()` → `visualize(ep)` | pure matplotlib |

### Key classes

- **`EpisodeData`** (line 48): parsed NPZ with typed attributes (path, steps, image_key, action_key, task, success, arrays)
- **`RoboViewApp`** (line 555): the main analysis GUI — tkinter `Tk` window with a sidebar (file list, playback controls, flip checkboxes, info panel) and a workspace containing an embedded `matplotlib.figure.Figure` via `FigureCanvasTkAgg`
- **`SelectedEpisodes`** (line 59): selectable episode list for the legacy import window

### Image flip (horizontal + vertical)

The `_normalize_image()` function (line 246) applies flips in this order:
1. `np.flipud(img)` if `flip_vertical=True` (default both flips on)
2. `np.fliplr(img)` if `flip_horizontal=True`

Both are controlled by separate checkboxes in both the import window toolbar and the analysis window sidebar. CLI equivalents:
```bash
python npz_visualizer.py demos --episode-index 1 --no-flip-horizontal --no-flip-vertical
```

### Fullscreen mode (RoboViewApp only)

Triggered by F11 / Escape / "全屏" button in the playback controls. Implementation:
- Enter: `_main_pane.forget(self._sidebar)` → hide sidebar → `self.root.state("zoomed")` → maximize window → canvas Configure event auto-resizes the Figure to fill the available space
- Exit: restore sidebar with `_main_pane.insert(0, ...)` → restore saved geometry → `self.root.state("normal")`
- F11/Escape use `bind_all()` (not `bind()`) to capture events regardless of which widget has focus (matplotlib canvas/toolbar would otherwise consume them)
- `_on_canvas_configure()` has a `_configure_guard` re-entrancy lock and a 0.3-inch deadband to avoid resize feedback loops

### Image-only mode (RoboViewApp only)

Toggled by the "仅图像" / "全部" button below the flip checkboxes. Implementation:
- `_toggle_image_only()` flips `self._image_only` and calls `_render_episode()`
- `_render_image_only()`: single `ax_img` (111) filling the entire Figure with `subplots_adjust(left=0, right=1, bottom=0, top=0.95)` — all curve subplots, cursors, and value text are absent
- `_render_full()`: the default 3×3 grid layout (image + 6 curve plots + value text)
- `_refresh_frame()` guards against None `value_artist` via `if self.value_artist is not None:` (was `hasattr` before image-only mode was added)

### Keyboard shortcuts (shared)

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| Left / Right | Frame back / forward |
| Home / End | First / Last frame |
| F11 | Toggle fullscreen (RoboViewApp) / matplotlib fullscreen (visualize) |
| Escape | Exit fullscreen (RoboViewApp only) |

## 采集 IK 后端与手感（2026-09-01，性能实测 2026-09-02）

- **必须用 `--ik-backend mujoco` 采集**：lerobot 后端执行偏差 30-50cm，操作员视觉补偿会被固化进 label（label 比 MuJoCo 实际位移少 ~13%，中位比值 0.87），推理用 mujoco 后端精确执行 label → 必然欠程。**采集前后跑 `check_label_vs_eef.py` 验证比值 0.9-1.1**
- **`--ik-tol`（默认 1e-3）**：mujoco 后端 least_squares 容差。⚠️ 实测只快 ~9%（nfev 本来就 3-5 次，tol 松紧几乎不改变迭代数）——**不是性能主战场**。`DemoCollector.__init__` 里必须 `self.ik_tol = ik_tol`（曾漏赋值）
- **性能实测分解（`libero-unity/test/benchmark_ik_perf.py`，2026-09-02）**：
  - 每帧每臂成本：so100 翻译层 **1.43ms → 0.85ms**（ftol 1e-8→1e-2 后）+ MuJoCo arm_ik.ik 0.77-0.82ms（1e-3）。lerobot 后端总计 0.89ms、mujoco 1.72ms
  - **翻译层 `so100_ik` 曾是最大单项且两后端共有**——lerobot 后端**不是解析解**，它唯一的成本就是翻译层数值 LS（原 ftol 硬编码 1e-8）。翻译解只作 MuJoCo FK/IK 目标，0.03mm 误差无感 → `so100_chain.so100_ik(ftol=1e-2)` 已默认放宽
  - 稳态瓶颈 = 每次 least_squares 迭代的固定开销 ~250-370µs（scipy TRF 有限差分 + numpy 小矩阵），不是迭代次数
  - **偶发 10-16ms 卡帧根因 = Python GC**（scipy 每帧小对象触发 gen1/2 收集）；`collect_datasets.py run()` 已 `gc.disable()`（实测卡帧 max 16.5→5.4ms；对象由引用计数即时释放，无循环引用，安全）
  - CPU 计算，**换显卡无效**
- **摇杆输入是开关式恒速**（joyconrobotics `common_update()`）：>4000 全速 0.1 m/s / <1000 反向 / **1000-4000 死区**——"推好久才动"部分是死区 + 跟随暂态，非比例控制。手腕 roll/pitch 另有低通滤波（α=0.08，~0.2s 滞后）
- 手感仍重可调：Unity `MjJoyConController.MaxJointDelta 0.3→0.6`、XML `kp 50→80 / forcerange 3.5→5.0`（XML 未被 git 跟踪，改前备份）

## Known issues

- **Joy-Con Z drift**: Z axis produces tiny per-frame drift (~1e-4 m) that accumulates but the robot doesn't execute. Consider a dead zone threshold in `_compute_action_label()` if it affects training.
- **Step() no-op in MuJoCo mode**: During data collection, physics is driven by `MjJoyConController` via JoyConReceiver (5555), NOT via `Step()`. The `env.step()` path is only for inference — and only works if Unity has `RobotArmController` instances (requires BDDL scene).
- **Image letterbox**: Camera renders at 16:9 → letterboxed to 224×224 square with black bars top/bottom (`ObservationCollector.cs`). No Python-side changes needed.
- **Old demo data quality (fixed 2026-07-28)**: Demos collected before 2026-07-28 have `robot0_joint_pos`=0, `robot0_gripper_qpos`=0, and stale `robot0_eef_pos` due to broken `FindJointId` in Unity. Only `action` (EEF delta) and `agentview_image` (video) are correct. Re-collect with fixed code for complete data.
