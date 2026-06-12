# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the data collection side of a dual-arm robotic block-grasping system. It uses Joy-Con teleop to control SO100 robots in Unity via TCP, records (observation, action) trajectories as `.npz` files, and provides format conversion for training.

The Unity simulator lives at `C:\vla\libero-unity` and the ML training side at `C:\vla\network`.

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
- `gripper`: absolute gripper target (0=open, 1=closed)

**Action order in 14-dim array**: `[R_dx,R_dy,R_dz,R_dRx,R_dRy,R_dRz,R_grip, L_dx,L_dy,L_dz,L_dRx,L_dRy,L_dRz,L_grip]`

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

Computes joint angles via `lerobot_IK`:
```python
y_r = 0.01                         # fixed lateral offset
pitch_r = -pitch_r
roll_r = roll_r - math.pi / 2
right_target_gpos = [x_r, y_r, z_r, roll_r, pitch_r, 0.0]  # yaw=0; yaw is handled by base joint
# → lerobot_IK() → joints
# → Base yaw: yaw_r → J0 (target_qpos[0])
```

**IMPORTANT**: `[x_r, y_r, z_r]` is in a LOCAL cylindrical coordinate frame that rotates with the base joint (yaw). It is NOT the same as the MuJoCo world-frame `obs["robot0_eef_pos"]`.

### Helper: `_euler_zxy_to_quat(rx, ry, rz)`

Converts extrinsic Z-X-Y Euler angles to quaternion [x, y, z, w], matching Unity's `Quaternion.Euler(rx, ry, rz)` convention. Quaternion formula: `q = q_y ⊗ q_x ⊗ q_z`.

### Home reset (`_go_home`)

On A/Y button press, resets IK warm-start to `_INIT_ARM_Q`, computes home joint angles via IK, sends via JoyConReceiver, and resets Joy-Con internal state:
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

## Known issues

- **Joy-Con Z drift**: Z axis produces tiny per-frame drift (~1e-4 m) that accumulates but the robot doesn't execute. Consider a dead zone threshold in `_compute_action_label()` if it affects training.
- **Step() no-op in MuJoCo mode**: During data collection, physics is driven by `MjJoyConController` via JoyConReceiver (5555), NOT via `Step()`. The `env.step()` path is only for inference — and only works if Unity has `RobotArmController` instances (requires BDDL scene).
- **Image letterbox**: Camera renders at 16:9 → letterboxed to 224×224 square with black bars top/bottom (`ObservationCollector.cs`). No Python-side changes needed.
