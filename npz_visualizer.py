#!/usr/bin/env python3
"""Visualize Unity robot demonstration episodes stored as NPZ files.

The collector in prompt.md saves one episode per .npz with fields such as:
agentview_image, robot0_joint_pos, robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos, robot1_joint_pos, robot1_eef_pos, robot1_eef_quat,
robot1_gripper_qpos, action, language_instruction, success.

This tool focuses on quick inspection:
  python npz_visualizer.py
  python npz_visualizer.py episode.npz
  python npz_visualizer.py episode.npz --info-only
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_LABELS = [
    "r_dx",
    "r_dy",
    "r_dz",
    "r_dRx",
    "r_dRy",
    "r_dRz",
    "r_grip",
    "l_dx",
    "l_dy",
    "l_dz",
    "l_dRx",
    "l_dRy",
    "l_dRz",
    "l_grip",
]


@dataclass
class EpisodeData:
    path: Path
    arrays: dict[str, Any]
    steps: int
    image_key: str | None
    action_key: str | None
    task: str
    success: Any


@dataclass
class SelectedEpisodes:
    episode: EpisodeData
    files: list[Path]


def _scalar_to_python(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _shape_text(value: Any) -> str:
    arr = np.asarray(value)
    return str(tuple(arr.shape)) if arr.shape else "scalar"


def _dtype_text(value: Any) -> str:
    return str(np.asarray(value).dtype)


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    value = _scalar_to_python(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_existing(keys: Iterable[str], arrays: dict[str, Any]) -> str | None:
    for key in keys:
        if key in arrays:
            return key
    return None


def _looks_like_image_sequence(key: str, value: Any) -> bool:
    arr = np.asarray(value)
    if arr.ndim < 3:
        return False
    lowered = key.lower()
    if any(token in lowered for token in ("image", "rgb", "camera", "agentview")):
        return True
    return arr.ndim in (3, 4) and arr.shape[-1] in (1, 3, 4)


def _infer_steps(arrays: dict[str, Any], action_key: str | None, image_key: str | None) -> int:
    if action_key is not None:
        action = np.asarray(arrays[action_key])
        if action.ndim >= 1:
            return int(action.shape[0])
    if image_key is not None:
        image = np.asarray(arrays[image_key])
        if image.ndim >= 1:
            return int(image.shape[0])

    candidates: list[int] = []
    for value in arrays.values():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] > 1:
            candidates.append(int(arr.shape[0]))
    if not candidates:
        return 0
    return max(set(candidates), key=candidates.count)


def load_episode(path: str | os.PathLike[str]) -> EpisodeData:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        arrays = {key: npz[key] for key in npz.files}

    action_key = _first_existing(["action", "actions"], arrays)
    image_key = _first_existing(["agentview_image", "image", "images", "rgb"], arrays)
    if image_key is None:
        for key, value in arrays.items():
            if _looks_like_image_sequence(key, value):
                image_key = key
                break

    task = _as_str(arrays.get("language_instruction"), default="")
    if not task:
        task = _as_str(arrays.get("task"), default="unknown")

    success = _scalar_to_python(arrays["success"]) if "success" in arrays else "unknown"
    steps = _infer_steps(arrays, action_key, image_key)
    return EpisodeData(npz_path, arrays, steps, image_key, action_key, task, success)


def concatenate_episodes(files: list[Path]) -> EpisodeData:
    episodes = [load_episode(path) for path in files]
    if not episodes:
        raise ValueError("No episodes to concatenate.")

    image_key = episodes[0].image_key
    action_key = episodes[0].action_key
    if any(ep.image_key != image_key for ep in episodes):
        raise ValueError("Cannot concatenate episodes with different image keys.")
    if any(ep.action_key != action_key for ep in episodes):
        raise ValueError("Cannot concatenate episodes with different action keys.")

    common_keys = set(episodes[0].arrays)
    for ep in episodes[1:]:
        common_keys.intersection_update(ep.arrays)

    arrays: dict[str, Any] = {}
    total_steps = sum(ep.steps for ep in episodes)
    for key in sorted(common_keys):
        chunks: list[np.ndarray] = []
        can_concat = True
        tail_shape: tuple[int, ...] | None = None
        for ep in episodes:
            arr = np.asarray(ep.arrays[key])
            if arr.ndim == 0 or arr.shape[0] != ep.steps:
                can_concat = False
                break
            if tail_shape is None:
                tail_shape = arr.shape[1:]
            elif arr.shape[1:] != tail_shape:
                can_concat = False
                break
            chunks.append(arr)
        if can_concat and chunks:
            arrays[key] = np.concatenate(chunks, axis=0)

    arrays["episode_id"] = np.concatenate(
        [np.full(ep.steps, index, dtype=np.int32) for index, ep in enumerate(episodes, start=1)]
    )
    arrays["episode_local_step"] = np.concatenate(
        [np.arange(ep.steps, dtype=np.int32) for ep in episodes]
    )
    arrays["language_instruction"] = np.array(episodes[0].task)
    arrays["success"] = np.array(all(bool(ep.success) for ep in episodes))

    merged_name = f"{files[0].parent.name}_merged_{len(files)}_episodes.npz"
    merged_path = files[0].parent / merged_name
    task = f"{episodes[0].task} (merged {len(files)} episodes)"
    return EpisodeData(merged_path, arrays, total_steps, image_key, action_key, task, arrays["success"].item())


def find_npz_files(path: str | os.PathLike[str]) -> list[Path]:
    input_path = Path(path).expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in directory: {input_path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {input_path}")


def print_episode_list(files: list[Path]) -> None:
    print(f"Found {len(files)} episode(s):")
    for index, path in enumerate(files, start=1):
        try:
            ep = load_episode(path)
            print(
                f"  {index:02d}. {path.name}  "
                f"steps={ep.steps} task={ep.task} success={ep.success}"
            )
        except Exception as exc:
            print(f"  {index:02d}. {path.name}  ERROR: {exc}")


def print_info(ep: EpisodeData) -> None:
    print(f"File: {ep.path}")
    print(f"Task: {ep.task}")
    print(f"Success: {ep.success}")
    print(f"Steps: {ep.steps}")
    print(f"Image key: {ep.image_key or 'not found'}")
    print(f"Action key: {ep.action_key or 'not found'}")
    print()
    print("Arrays:")
    width = max((len(key) for key in ep.arrays), default=4)
    for key in sorted(ep.arrays):
        value = ep.arrays[key]
        print(f"  {key:<{width}}  shape={_shape_text(value):<18} dtype={_dtype_text(value)}")


def _normalize_image(frame: np.ndarray, flip_vertical: bool = True) -> np.ndarray:
    img = np.asarray(frame)
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    elif img.ndim != 3:
        img = np.squeeze(img)

    if flip_vertical and img.ndim >= 2:
        img = np.flipud(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32, copy=False)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.float32)

    min_v = float(np.nanmin(img))
    max_v = float(np.nanmax(img))
    if min_v >= 0.0 and max_v <= 1.0:
        return np.clip(img, 0.0, 1.0)
    if min_v >= 0.0 and max_v <= 255.0:
        return np.clip(img / 255.0, 0.0, 1.0)
    if math.isclose(min_v, max_v):
        return np.zeros(img.shape, dtype=np.float32)
    return np.clip((img - min_v) / (max_v - min_v), 0.0, 1.0)


def _series(ep: EpisodeData, key: str) -> np.ndarray | None:
    if key not in ep.arrays:
        return None
    arr = np.asarray(ep.arrays[key])
    if arr.ndim == 0:
        return None
    if ep.steps and arr.shape[0] != ep.steps:
        arr = arr[: ep.steps]
    return arr.reshape(arr.shape[0], -1)


def _format_values(name: str, values: np.ndarray | None, frame: int, max_items: int = 6) -> str:
    if values is None or values.size == 0:
        return f"{name}: n/a"
    row = values[min(frame, values.shape[0] - 1)]
    shown = ", ".join(f"{v:+.4f}" for v in row[:max_items])
    if row.size > max_items:
        shown += ", ..."
    return f"{name}: [{shown}]"


def _plot_multidim(ax: Any, y: np.ndarray | None, labels: list[str], title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    if y is None or y.size == 0:
        ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
        return
    for idx in range(y.shape[1]):
        label = labels[idx] if idx < len(labels) else f"{idx}"
        ax.plot(y[:, idx], linewidth=1.0, label=label)
    if y.shape[1] <= 14:
        ax.legend(loc="upper right", ncols=2, fontsize=8)


def _preferred_cjk_font() -> Any | None:
    try:
        from matplotlib import font_manager
    except Exception:
        return None

    preferred = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS")
    by_name = {font.name: font.fname for font in font_manager.fontManager.ttflist}
    for name in preferred:
        path = by_name.get(name)
        if path:
            return font_manager.FontProperties(fname=path)
    return None


def visualize(ep: EpisodeData, fps: float = 30.0, flip_image: bool = True) -> None:
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    if ep.steps <= 0:
        raise ValueError("No time dimension could be inferred from this NPZ file.")

    cjk_font = _preferred_cjk_font()

    action = _series(ep, ep.action_key) if ep.action_key else None
    robot0_joint = _series(ep, "robot0_joint_pos")
    robot1_joint = _series(ep, "robot1_joint_pos")
    robot0_eef_pos = _series(ep, "robot0_eef_pos")
    robot1_eef_pos = _series(ep, "robot1_eef_pos")
    robot0_eef_quat = _series(ep, "robot0_eef_quat")
    robot1_eef_quat = _series(ep, "robot1_eef_quat")
    robot0_grip = _series(ep, "robot0_gripper_qpos")
    robot1_grip = _series(ep, "robot1_gripper_qpos")
    episode_id = _series(ep, "episode_id")
    episode_local_step = _series(ep, "episode_local_step")

    images = np.asarray(ep.arrays[ep.image_key]) if ep.image_key else None
    has_images = images is not None and images.ndim >= 3 and images.shape[0] > 0

    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title(f"NPZ Visualizer - {ep.path.name}")
    gs = fig.add_gridspec(5, 3, height_ratios=[1.2, 1.0, 1.0, 0.18, 0.18], width_ratios=[1.1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_text = fig.add_subplot(gs[2, 0])
    ax_action_pos = fig.add_subplot(gs[0, 1])
    ax_action_rot = fig.add_subplot(gs[0, 2])
    ax_joint = fig.add_subplot(gs[1, 1])
    ax_eef_pos = fig.add_subplot(gs[1, 2])
    ax_eef_quat = fig.add_subplot(gs[2, 1])
    ax_grip = fig.add_subplot(gs[2, 2])
    ax_slider = fig.add_subplot(gs[3, :])
    ax_speed = fig.add_subplot(gs[4, :])

    if has_images:
        first_image = _normalize_image(images[0], flip_vertical=flip_image)
        image_artist = ax_img.imshow(first_image)
        ax_img.set_title(ep.image_key)
    else:
        image_artist = None
        ax_img.text(0.5, 0.5, "image key not found", ha="center", va="center", transform=ax_img.transAxes)
        ax_img.set_title("camera")
    ax_img.axis("off")

    action_pos = action[:, [0, 1, 2, 7, 8, 9]] if action is not None and action.shape[1] >= 10 else action
    action_rot = action[:, [3, 4, 5, 10, 11, 12]] if action is not None and action.shape[1] >= 13 else None
    _plot_multidim(ax_action_pos, action_pos, ["r_dx", "r_dy", "r_dz", "l_dx", "l_dy", "l_dz"], "Action position deltas")
    _plot_multidim(ax_action_rot, action_rot, ["r_dRx", "r_dRy", "r_dRz", "l_dRx", "l_dRy", "l_dRz"], "Action rotation deltas")

    joint_series: list[np.ndarray] = []
    joint_labels: list[str] = []
    if robot0_joint is not None:
        joint_series.append(robot0_joint)
        joint_labels.extend([f"r_j{i}" for i in range(robot0_joint.shape[1])])
    if robot1_joint is not None:
        joint_series.append(robot1_joint)
        joint_labels.extend([f"l_j{i}" for i in range(robot1_joint.shape[1])])
    joint = np.concatenate(joint_series, axis=1) if joint_series else None
    _plot_multidim(ax_joint, joint, joint_labels, "Robot joint positions")

    eef_pos_series: list[np.ndarray] = []
    eef_pos_labels: list[str] = []
    if robot0_eef_pos is not None:
        eef_pos_series.append(robot0_eef_pos)
        eef_pos_labels.extend(["r_x", "r_y", "r_z"][: robot0_eef_pos.shape[1]])
    if robot1_eef_pos is not None:
        eef_pos_series.append(robot1_eef_pos)
        eef_pos_labels.extend(["l_x", "l_y", "l_z"][: robot1_eef_pos.shape[1]])
    eef_pos = np.concatenate(eef_pos_series, axis=1) if eef_pos_series else None
    _plot_multidim(ax_eef_pos, eef_pos, eef_pos_labels, "End-effector positions")

    eef_quat_series: list[np.ndarray] = []
    eef_quat_labels: list[str] = []
    if robot0_eef_quat is not None:
        eef_quat_series.append(robot0_eef_quat)
        eef_quat_labels.extend(["r_qx", "r_qy", "r_qz", "r_qw"][: robot0_eef_quat.shape[1]])
    if robot1_eef_quat is not None:
        eef_quat_series.append(robot1_eef_quat)
        eef_quat_labels.extend(["l_qx", "l_qy", "l_qz", "l_qw"][: robot1_eef_quat.shape[1]])
    eef_quat = np.concatenate(eef_quat_series, axis=1) if eef_quat_series else None
    _plot_multidim(ax_eef_quat, eef_quat, eef_quat_labels, "End-effector quaternions")

    grip_series: list[np.ndarray] = []
    grip_labels: list[str] = []
    if robot0_grip is not None:
        grip_series.append(robot0_grip)
        grip_labels.extend([f"r_grip{i}" for i in range(robot0_grip.shape[1])])
    if robot1_grip is not None:
        grip_series.append(robot1_grip)
        grip_labels.extend([f"l_grip{i}" for i in range(robot1_grip.shape[1])])
    if action is not None and action.shape[1] >= 14:
        grip_series.append(action[:, [6, 13]])
        grip_labels.extend(["r_action_grip", "l_action_grip"])
    grip = np.concatenate(grip_series, axis=1) if grip_series else None
    _plot_multidim(ax_grip, grip, grip_labels, "Gripper values")

    plot_axes = [ax_action_pos, ax_action_rot, ax_joint, ax_eef_pos, ax_eef_quat, ax_grip]
    cursors = [ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.75) for ax in plot_axes]

    ax_text.axis("off")
    text_kwargs = {"fontproperties": cjk_font} if cjk_font is not None else {"family": "monospace"}
    text_artist = ax_text.text(0.0, 1.0, "", va="top", fontsize=9, **text_kwargs)

    slider = Slider(ax_slider, "step", 0, ep.steps - 1, valinit=0, valstep=1)
    speed_slider = Slider(ax_speed, "fps", 1.0, 120.0, valinit=float(fps), valstep=1.0)
    state = {
        "frame": 0,
        "playing": False,
        "last_tick": time.monotonic(),
        "frame_accum": 0.0,
        "fps": float(fps),
    }

    btn_ax = fig.add_axes([0.012, 0.012, 0.08, 0.035])
    play_button = Button(btn_ax, "Play")

    def update(frame: int) -> None:
        frame = int(np.clip(frame, 0, ep.steps - 1))
        state["frame"] = frame
        if image_artist is not None and has_images:
            image_artist.set_data(_normalize_image(images[min(frame, images.shape[0] - 1)], flip_vertical=flip_image))
        for cursor in cursors:
            cursor.set_xdata([frame, frame])
        episode_line = ""
        if episode_id is not None and episode_local_step is not None:
            ep_num = int(episode_id[min(frame, episode_id.shape[0] - 1), 0])
            local_step = int(episode_local_step[min(frame, episode_local_step.shape[0] - 1), 0])
            episode_line = f"episode: {ep_num}, local step: {local_step + 1}"
        text_lines = [
            f"file: {ep.path.name}",
            f"task: {ep.task}",
            f"success: {ep.success}",
            f"step: {frame + 1}/{ep.steps}",
            episode_line,
            "",
            _format_values("action", action, frame, 14),
            _format_values("r_joint", robot0_joint, frame),
            _format_values("l_joint", robot1_joint, frame),
            _format_values("r_eef_pos", robot0_eef_pos, frame, 3),
            _format_values("l_eef_pos", robot1_eef_pos, frame, 3),
            _format_values("r_eef_quat", robot0_eef_quat, frame, 4),
            _format_values("l_eef_quat", robot1_eef_quat, frame, 4),
        ]
        text_artist.set_text("\n".join(text_lines))
        fig.canvas.draw_idle()

    def on_slider(value: float) -> None:
        state["frame_accum"] = 0.0
        update(int(value))

    def on_speed(value: float) -> None:
        state["fps"] = max(1.0, float(value))

    def on_play(_event: Any) -> None:
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        state["last_tick"] = time.monotonic()
        state["frame_accum"] = 0.0

    def on_key(event: Any) -> None:
        if event.key in (" ", "space"):
            on_play(event)
        elif event.key in ("right", "d"):
            slider.set_val(min(ep.steps - 1, state["frame"] + 1))
        elif event.key in ("left", "a"):
            slider.set_val(max(0, state["frame"] - 1))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(ep.steps - 1)

    slider.on_changed(on_slider)
    speed_slider.on_changed(on_speed)
    play_button.on_clicked(on_play)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)

    timer = fig.canvas.new_timer(interval=16)

    def tick() -> None:
        if state["playing"]:
            now = time.monotonic()
            elapsed = now - state["last_tick"]
            state["last_tick"] = now
            state["frame_accum"] += elapsed * state["fps"]
            step_count = int(state["frame_accum"])
            if step_count > 0:
                state["frame_accum"] -= step_count
                next_frame = state["frame"] + step_count
                if next_frame >= ep.steps:
                    next_frame %= ep.steps
                slider.set_val(next_frame)

    timer.add_callback(tick)
    timer.start()

    title_kwargs = {"fontproperties": cjk_font} if cjk_font is not None else {}
    fig.suptitle(f"{ep.path.name}  |  {ep.task}", fontsize=12, **title_kwargs)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.105, top=0.93, wspace=0.28, hspace=0.70)
    plt.show()


def launch_import_app(default_dir: str = "demos") -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("RoboView NPZ Visualizer")
    root.geometry("920x560")

    files: list[Path] = []
    selected_file = tk.StringVar(value="")
    fps_var = tk.DoubleVar(value=30.0)
    flip_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="导入一个 .npz 文件，或导入包含 .npz 的文件夹。")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    toolbar = ttk.Frame(root, padding=10)
    toolbar.grid(row=0, column=0, sticky="ew")
    toolbar.columnconfigure(5, weight=1)

    list_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    list_frame.grid(row=1, column=0, sticky="nsew")
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)

    bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
    bottom.grid(row=2, column=0, sticky="ew")
    bottom.columnconfigure(0, weight=1)

    file_list = tk.Listbox(list_frame, height=14)
    file_list.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=file_list.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    file_list.configure(yscrollcommand=scrollbar.set)

    info_text = tk.Text(bottom, height=8, wrap="none")
    info_text.grid(row=0, column=0, sticky="ew")
    info_text.configure(state="disabled")

    def set_info(text: str) -> None:
        info_text.configure(state="normal")
        info_text.delete("1.0", "end")
        info_text.insert("1.0", text)
        info_text.configure(state="disabled")

    def refresh_list(new_files: list[Path]) -> None:
        files.clear()
        files.extend(new_files)
        file_list.delete(0, "end")
        for index, path in enumerate(files, start=1):
            try:
                ep = load_episode(path)
                label = f"{index:02d}. {path.name}    steps={ep.steps}    image={ep.arrays[ep.image_key].shape if ep.image_key else 'n/a'}"
            except Exception as exc:
                label = f"{index:02d}. {path.name}    ERROR: {exc}"
            file_list.insert("end", label)
        if files:
            file_list.selection_set(0)
            file_list.activate(0)
            selected_file.set(str(files[0]))
            show_selected_info()
        status_var.set(f"已导入 {len(files)} 个 .npz 文件。")

    def get_selected_path() -> Path | None:
        selection = file_list.curselection()
        if not selection:
            if len(files) == 1:
                return files[0]
            return None
        return files[int(selection[0])]

    def show_selected_info(_event: Any = None) -> None:
        path = get_selected_path()
        if path is None:
            set_info("")
            return
        try:
            ep = load_episode(path)
            lines = [
                f"文件: {ep.path}",
                f"任务: {ep.task}",
                f"成功: {ep.success}",
                f"步数: {ep.steps}",
                f"图像: {ep.image_key or 'not found'}",
                f"动作: {ep.action_key or 'not found'}",
                "",
                "数组:",
            ]
            width = max((len(key) for key in ep.arrays), default=4)
            for key in sorted(ep.arrays):
                value = ep.arrays[key]
                lines.append(f"  {key:<{width}}  shape={_shape_text(value):<18} dtype={_dtype_text(value)}")
            set_info("\n".join(lines))
            selected_file.set(str(path))
        except Exception as exc:
            set_info(f"读取失败: {exc}")

    def open_npz() -> None:
        path = filedialog.askopenfilename(
            title="导入 NPZ 文件",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            initialdir=str(Path(default_dir).resolve()) if Path(default_dir).exists() else str(Path.cwd()),
        )
        if path:
            refresh_list([Path(path)])

    def open_folder() -> None:
        path = filedialog.askdirectory(
            title="导入 NPZ 文件夹",
            initialdir=str(Path(default_dir).resolve()) if Path(default_dir).exists() else str(Path.cwd()),
        )
        if not path:
            return
        try:
            refresh_list(find_npz_files(path))
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def analyze_selected() -> None:
        path = get_selected_path()
        if path is None:
            messagebox.showwarning("未选择文件", "请先导入并选择一个 .npz 文件。")
            return
        try:
            ep = load_episode(path)
            print_info(ep)
            visualize(ep, fps=fps_var.get(), flip_image=flip_var.get())
        except Exception as exc:
            messagebox.showerror("分析失败", str(exc))

    ttk.Button(toolbar, text="导入 NPZ", command=open_npz).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(toolbar, text="导入文件夹", command=open_folder).grid(row=0, column=1, padx=(0, 16))
    ttk.Button(toolbar, text="开始分析", command=analyze_selected).grid(row=0, column=2, padx=(0, 16))
    ttk.Label(toolbar, text="播放速度 fps").grid(row=0, column=3, padx=(0, 6))
    ttk.Scale(toolbar, from_=1, to=120, variable=fps_var, orient="horizontal", length=180).grid(row=0, column=4)
    ttk.Label(toolbar, textvariable=fps_var, width=6).grid(row=0, column=5, sticky="w", padx=(6, 16))
    ttk.Checkbutton(toolbar, text="垂直翻转图像", variable=flip_var).grid(row=0, column=6, padx=(0, 8))

    ttk.Label(bottom, textvariable=status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
    file_list.bind("<<ListboxSelect>>", show_selected_info)

    if Path(default_dir).exists():
        try:
            refresh_list(find_npz_files(default_dir))
        except Exception:
            pass

    root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize robot demonstration NPZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to one episode .npz file or a directory containing .npz files.",
    )
    parser.add_argument("--app", action="store_true", help="Open the import-and-analyze desktop UI.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="1-based episode index to open when path is a directory.",
    )
    parser.add_argument("--concat-all", action="store_true", help="Merge all episodes in a directory into one timeline.")
    parser.add_argument("--no-flip-image", action="store_true", help="Do not vertically flip camera frames.")
    parser.add_argument("--list", action="store_true", help="List episodes and exit.")
    parser.add_argument("--info-only", action="store_true", help="Only print metadata; do not open the GUI.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback speed for the GUI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        wants_plain_app = (
            args.path is None
            and not args.list
            and not args.info_only
            and not args.concat_all
            and args.episode_index is None
            and not args.no_flip_image
            and float(args.fps) == 30.0
        )
        if args.app or wants_plain_app:
            launch_import_app()
            return 0

        input_path = args.path or "demos"
        files = find_npz_files(input_path)
        if args.list:
            print_episode_list(files)
            return 0

        should_concat = args.concat_all

        if should_concat:
            ep = concatenate_episodes(files)
        else:
            episode_index = args.episode_index or 1
            if episode_index < 1 or episode_index > len(files):
                raise ValueError(f"--episode-index must be between 1 and {len(files)}")
            ep = load_episode(files[episode_index - 1])
        print_info(ep)
        if not args.info_only:
            visualize(ep, fps=args.fps, flip_image=not args.no_flip_image)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
