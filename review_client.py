#!/usr/bin/env python3
"""RECAP 成功候选 —— 逐集人工审阅客户端（可视化 + 落盘决策）。

背景
----
`collect_rollouts.py` 只把自动判据（MuJoCo 真值：末态 XY 在盒心 0.07 m 内、红块
比 reset 初态抬高 ≥12 mm、且抓取后发生过张爪）通过的 episode 标成 `pending`，
**永远不会自动进训练**。本客户端把这一步做成逐集可视化判断：看一眼（可回放整段）、
按 A/R/S 决定批准/拒绝/跳过，决策追加写入 JSONL，格式与 `recap/review_rollouts.py`
完全一致，因此下游 `recap/finalize_reviews.py` 无需任何改动。

关键方向约定（踩过坑）
----------------------
`DatasetsCollector` 采集的示教 npz 存的是 **rot180** 后的图（`env_client.py:133`），
而 RECAP rollout（`recap/rollout_format.py`）存的是 **原始正向图**。
两者方向相反 —— 故本工具默认 **不翻转**（`--flip` 才翻转，用于示教格式）。

用法
----
    # 只看候选队列与素材是否齐全（不开窗，可脚本化）
    python review_client.py --manifest review/round-004-objpos/manifest.jsonl --check

    # 正式审阅（有 raw npz：可回放整段 + 画红块轨迹）
    python review_client.py --manifest ... \\ 
        --decisions review/round-004-objpos/decisions.jsonl \\
        --reviewer wys --raw-dir review/round-004-objpos/raw

    # 只有 final/min 两张关键帧时（画不出曲线，仍可判断）
    python review_client.py --manifest ... --decisions ... --reviewer wys \\
        --frames-dir review/round-004-objpos/frames

按键
----
    Space        播放/暂停          ←/→   上一帧/下一帧
    Home/End     首帧/末帧          A     批准    R  拒绝    S  跳过
    U            撤销上一集决策     P/N   上一集/下一集
    Q / Esc      退出

进度条
------
    图像下方的进度条可**拖动**：按住任意位置左右拖，实时定位到对应帧。
    点击任意位置可直接跳转，旁边显示当前帧和百分比。
    播放中拖动会先暂停、松手后继续，避免播放与滑块互相抢帧。没有 raw npz（只能看静态
    关键帧）时滑块自动禁用并显示「无 raw，不可拖动」。

输出
----
decisions JSONL，每行：
    {"episode_id": ..., "decision": "approved|rejected|skip",
     "reviewer": ..., "reviewed_at": "<UTC ISO8601>"}
只追加、可续（已判过的 episode 自动跳过）。`U` 撤销采用**重写文件**而非追加反向
记录 —— 因为 `finalize_reviews.py` 遇到重复 episode_id 会直接报错。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from npz_visualizer import (  # noqa: E402  (复用既有加载/归一化/字体)
    EpisodeData,
    _normalize_image,
    _preferred_cjk_font,
    load_episode,
)

# ── 判据常量：与 recap/../run_inference.py 保持一致（改动需两处同步）──────
BOX_CX, BOX_CY = -0.42, 0.02     # box_1 中心（MuJoCo world）
IN_BOX_R = 0.07                  # 末态红块 XY 离盒心上限（m）
MIN_BOX_LIFT = 0.012             # 红块需比 reset 初态抬高（m）
RED_BLOCK = "red_block_1"
FRAME_SUFFIXES = ("final", "min")

TITLE_COLOR = "#cde"
AXIS_COLOR = "#9ab"
BG = "#0b0d10"
SIDE_BG = "#15181c"


# ── 数据装载 ────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def load_decisions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        eid = row.get("episode_id")
        if eid:
            out[eid] = row
    return out


def append_decision(path: Path, episode_id: str, decision: str, reviewer: str) -> dict:
    record = {
        "episode_id": episode_id,
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
    return record


def pop_last_decision(path: Path) -> Optional[dict]:
    """撤销：重写 decisions 文件并去掉最后一行。

    不能追加一条反向记录 —— finalize_reviews.py 见到重复的 episode_id 会抛错。
    """
    if not path.is_file():
        return None
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(l + "\n" for l in lines[:-1]), encoding="utf-8")
    os.replace(tmp, path)
    return last


def find_raw(raw_dir: Optional[Path], episode_id: str) -> Optional[Path]:
    if not raw_dir or not raw_dir.is_dir():
        return None
    direct = raw_dir / f"{episode_id}.npz"
    if direct.is_file():
        return direct
    hits = sorted(raw_dir.glob(f"*{episode_id}*.npz"))
    return hits[0] if hits else None


def find_frames(frames_dir: Optional[Path], episode_id: str) -> list[tuple[str, Path]]:
    if not frames_dir or not frames_dir.is_dir():
        return []
    out = []
    for suffix in FRAME_SUFFIXES:
        path = frames_dir / f"{episode_id}_{suffix}.png"
        if path.is_file():
            out.append((suffix, path))
    return out


@dataclass
class Candidate:
    index: int
    row: dict
    status: str = "pending"

    episode: Optional[EpisodeData] = None
    raw_path: Optional[Path] = None
    frames: list[tuple[str, Path]] = field(default_factory=list)
    frames_pixels: list[tuple[str, np.ndarray]] = field(default_factory=list)
    dist: Optional[np.ndarray] = None      # (T,) 红块 XY 离盒心
    z: Optional[np.ndarray] = None         # (T,) 红块 z
    init_z: float = float("nan")
    jaw: Optional[np.ndarray] = None       # (T,) Jaw 观测
    error: str = ""

    @property
    def episode_id(self) -> str:
        return str(self.row.get("episode_id", f"#{self.index}"))

    @property
    def auto(self) -> dict:
        return self.row.get("auto") or {}

    @property
    def steps(self) -> int:
        if self.episode is not None and self.episode.steps:
            return int(self.episode.steps)
        if self.dist is not None:
            return int(self.dist.shape[0])
        return 1

    def raw_frame(self, frame: int) -> Optional[np.ndarray]:
        if self.episode is None or not self.episode.image_key:
            return None
        arr = np.asarray(self.episode.arrays[self.episode.image_key])
        if arr.ndim < 3 or arr.shape[0] == 0:
            return None
        return arr[min(max(0, frame), arr.shape[0] - 1)]


def build_candidate(cand: Candidate, raw_dir: Optional[Path], frames_dir: Optional[Path],
                    flip_h: bool, flip_v: bool) -> Candidate:
    """装载候选素材；任何失败记进 cand.error，不抛出（单坏文件不该打断审阅）。"""
    eid = cand.episode_id
    try:
        raw = find_raw(raw_dir, eid)
        if raw is not None:
            cand.raw_path = raw
            cand.episode = load_episode(raw)
            arrays = cand.episode.arrays
            if "object_positions" in arrays and "object_names" in arrays:
                names = [str(x) for x in np.asarray(arrays["object_names"]).ravel()]
                pos = np.asarray(arrays["object_positions"], dtype=np.float64)
                if RED_BLOCK in names and pos.ndim == 3 and pos.shape[1] > names.index(RED_BLOCK):
                    track = pos[:, names.index(RED_BLOCK), :]
                    if track.shape[1] >= 3:
                        cand.dist = np.hypot(track[:, 0] - BOX_CX, track[:, 1] - BOX_CY)
                        cand.z = track[:, 2]
                        # 初始高度必须与采集端判据同源：manifest 记的是 **reset 观测**
                        # 的 z（方块可能尚未落稳），而第一帧记录要晚 0.5 s（已稳定在桌面）
                        # —— 两者可差 ~6 mm，直接用首帧会让屏上的阈值线与记录的 dz 对不上。
                        recorded = (cand.auto or {}).get("initial_red_z")
                        cand.init_z = (float(recorded) if isinstance(recorded, (int, float))
                                       and np.isfinite(recorded) else float(track[0, 2]))
            jaw = arrays.get("robot0_joint_pos")
            if jaw is not None and np.asarray(jaw).ndim == 2 and np.asarray(jaw).shape[1] >= 6:
                cand.jaw = np.asarray(jaw, dtype=np.float64)[:, 5]
    except Exception as exc:  # noqa: BLE001
        cand.error = f"raw 装载失败: {type(exc).__name__}: {exc}"

    try:
        cand.frames = find_frames(frames_dir, eid)
        if cand.frames:
            from PIL import Image
            for suffix, path in cand.frames:
                img = np.asarray(Image.open(path).convert("RGB"))
                cand.frames_pixels.append((suffix, _normalize_image(img, flip_h, flip_v)))
    except Exception as exc:  # noqa: BLE001
        cand.error = (cand.error + " | " if cand.error else "") + f"关键帧装载失败: {exc}"
    return cand


def select_rows(manifest: Path, only: str) -> list[dict]:
    rows = read_jsonl(manifest)
    if only == "pending":
        rows = [r for r in rows if r.get("review_status") == "pending"]
    return rows


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):+.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


# ── GUI ─────────────────────────────────────────────────────────────────

class ReviewApp:
    """逐集审阅窗口。

    图形只建一次（axes + artist），切集时更新曲线数据、翻帧时只换图像数组 ——
    否则每帧重建 axes 会让回放掉到几 fps。
    """

    def __init__(self, args, candidates: list[Candidate], decisions: dict[str, dict]):
        import tkinter as tk
        from matplotlib import rcParams
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        font = _preferred_cjk_font()
        if font is not None:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [font.get_name()] + list(rcParams["font.sans-serif"])
            rcParams["axes.unicode_minus"] = False

        self.args = args
        self.candidates = candidates
        self.decisions_path = Path(args.decisions).expanduser()
        self.decisions = decisions
        for cand in candidates:
            rec = decisions.get(cand.episode_id)
            if rec:
                cand.status = rec["decision"]
        self.index = self._first_unjudged()
        self.frame = 0
        self.playing = False
        self._play_after_id = None
        self.fps = float(args.fps)
        self._syncing_scale = False      # 程序回写滑块时抑制 command，避免自激
        self._resume_after_scrub = False # 拖动前在播放 → 松手后恢复播放
        self._scrubbing = False
        self.raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None
        self.frames_dir = Path(args.frames_dir).expanduser() if args.frames_dir else None

        self.root = tk.Tk()
        self.root.title(f"RECAP 逐集审阅 — {len(candidates)} 个候选")
        self.root.geometry(args.geometry)

        side = tk.Frame(self.root, width=360, bg=SIDE_BG)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self.progress = tk.Label(side, text="", fg="#8fd", bg=SIDE_BG,
                                 font=("Consolas", 11, "bold"), anchor="w", justify="left")
        self.progress.pack(fill="x", padx=8, pady=(8, 4))
        self.listbox = tk.Listbox(side, bg="#0f1216", fg="#dde", selectbackground="#2b6cb0",
                                  font=("Consolas", 9), activestyle="none", height=12)
        self.listbox.pack(fill="both", expand=True, padx=8)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.info = tk.Label(side, text="", fg="#bcd", bg=SIDE_BG, justify="left",
                             anchor="nw", font=("Consolas", 9), wraplength=340)
        self.info.pack(fill="x", padx=8, pady=6)

        row1 = tk.Frame(side, bg=SIDE_BG)
        row1.pack(fill="x", padx=8, pady=2)
        for text, decision in (("批准 (A)", "approved"), ("拒绝 (R)", "rejected"), ("跳过 (S)", "skip")):
            tk.Button(row1, text=text, width=11,
                      command=lambda d=decision: self.judge(d)).pack(side="left", padx=2)
        row2 = tk.Frame(side, bg=SIDE_BG)
        row2.pack(fill="x", padx=8, pady=2)
        tk.Button(row2, text="撤销 (U)", width=11, command=self.undo).pack(side="left", padx=2)
        tk.Button(row2, text="上一集 (P)", width=11, command=lambda: self.step_case(-1)).pack(side="left", padx=2)
        tk.Button(row2, text="下一集 (N)", width=11, command=lambda: self.step_case(1)).pack(side="left", padx=2)
        row3 = tk.Frame(side, bg=SIDE_BG)
        row3.pack(fill="x", padx=8, pady=2)
        tk.Button(row3, text="播放/暂停 (Space)", command=self.toggle_play).pack(side="left", padx=2)
        tk.Label(side, text="A批准  R拒绝  S跳过  U撤销  P/N换集  Space播放  ←/→翻帧\n"
                            "拖动下方进度条可任意定位（播放中拖动会先暂停，松手继续）  Q退出",
                 fg="#678", bg=SIDE_BG, font=("Consolas", 8), wraplength=340,
                 justify="left").pack(side="bottom", fill="x", padx=8, pady=6)

        right = tk.Frame(self.root, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)

        scrub = tk.Frame(right, bg=BG)
        self.frame_label = tk.Label(scrub, text="", fg=AXIS_COLOR, bg=BG,
                                    font=("Consolas", 9), width=18, anchor="w")
        self.frame_label.pack(side="left")
        # 使用 Scale 的真实端点坐标定位；接管鼠标事件以支持从槽体任意位置开始拖动。
        self.scale = tk.Scale(scrub, from_=0, to=1, orient="horizontal", showvalue=False,
                              resolution=1, takefocus=1, bd=0, highlightthickness=0,
                              width=24, sliderlength=24,
                              bg="#ffffff", fg=AXIS_COLOR, troughcolor="#1b2027", sliderrelief="flat",
                              activebackground="#ffffff", command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True)
        self.scale.bind("<ButtonPress-1>", self._on_scale_press)
        self.scale.bind("<B1-Motion>", self._on_scale_motion)
        self.scale.bind("<ButtonRelease-1>", self._on_scale_release)

        self.status = tk.Label(right, text="", fg=AXIS_COLOR, bg=BG,
                               font=("Consolas", 9), anchor="w")
        self.status.pack(side="bottom", fill="x")
        scrub.pack(side="bottom", fill="x", padx=6, pady=(6, 4))
        self.frame_values = tk.Label(right, text="", fg=TITLE_COLOR, bg=BG,
                                     font=("Consolas", 11), anchor="w")
        self.frame_values.pack(side="bottom", fill="x", padx=8, pady=(4, 0))
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._build_figure()
        self._bind_keys()
        self._refresh_listbox()
        self.load_current()

    # ── 图形（只建一次）─────────────────────────────────────────────────
    def _build_figure(self):
        self.ax_img = self.figure.add_axes([0.02, 0.40, 0.96, 0.57])
        self.ax_curve = self.figure.add_axes([0.08, 0.07, 0.84, 0.28])
        self.ax_z = self.ax_curve.twinx()
        for ax in (self.ax_img, self.ax_curve, self.ax_z):
            ax.set_facecolor(BG)
            ax.tick_params(colors=AXIS_COLOR)
            for spine in ax.spines.values():
                spine.set_color("#334")
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.ax_img.set_title("", color=TITLE_COLOR, fontsize=10)
        self.ax_curve.set_xlabel("step", color=AXIS_COLOR)
        self.ax_curve.set_ylabel("红块 XY 离盒心 (m)", color="#4fc3f7")
        self.ax_curve.tick_params(axis="y", colors="#4fc3f7")
        self.ax_z.set_ylabel("红块 z (m)", color="#a5d6a7")
        self.ax_z.tick_params(axis="y", colors="#a5d6a7")
        self.ax_curve.grid(True, alpha=0.2, color="#345")

        self.im_artist = None
        self.text_artist = None
        self.dist_line, = self.ax_curve.plot([], [], color="#4fc3f7", linewidth=1.4,
                                             label="红块 XY 离盒心")
        self.in_box_line = self.ax_curve.axhline(IN_BOX_R, color="#ffb74d", linestyle="--",
                                                 linewidth=1.0, label=f"入盒阈值 {IN_BOX_R:.2f}")
        self.z_line, = self.ax_z.plot([], [], color="#a5d6a7", linewidth=1.2, label="红块 z")
        self.lift_line = self.ax_z.axhline(0.0, color="#81c784", linestyle=":", linewidth=1.0,
                                           label=f"初态+{MIN_BOX_LIFT:.3f}")
        self.jaw_line, = self.ax_z.plot([], [], color="#ba68c8", linewidth=0.8, alpha=0.7,
                                        label="Jaw×0.1")
        self.cursor = self.ax_curve.axvline(0, color="#fff", linewidth=0.8, alpha=0.6)

    def _set_curve_visibility(self, has_track: bool):
        for artist in (self.dist_line, self.in_box_line, self.z_line, self.lift_line,
                       self.jaw_line, self.cursor):
            artist.set_visible(has_track)

    # ── 导航 ────────────────────────────────────────────────────────────
    def _first_unjudged(self) -> int:
        for i, cand in enumerate(self.candidates):
            if cand.status == "pending":
                return i
        return 0

    def _current(self) -> Optional[Candidate]:
        if not self.candidates:
            return None
        return self.candidates[max(0, min(self.index, len(self.candidates) - 1))]

    def load_current(self):
        cand = self._current()
        if cand is None:
            self.status.configure(text="没有候选")
            return
        if cand.episode is None and not cand.frames_pixels and not cand.error:
            build_candidate(cand, self.raw_dir, self.frames_dir, self.args.flip, self.args.flip)
        self.frame = 0
        self._render_episode()

    def step_case(self, delta: int):
        if not self.candidates:
            return
        self.index = max(0, min(self.index + delta, len(self.candidates) - 1))
        self.load_current()
        self._refresh_listbox()

    # ── 决策 ────────────────────────────────────────────────────────────
    def judge(self, decision: str):
        cand = self._current()
        if cand is None:
            return
        append_decision(self.decisions_path, cand.episode_id, decision, self.args.reviewer)
        cand.status = decision
        nxt = self.index + 1
        while nxt < len(self.candidates) and self.candidates[nxt].status != "pending":
            nxt += 1
        if nxt < len(self.candidates):
            self.index = nxt
            self.load_current()
        else:
            self._update_info(cand)
            self.status.configure(text="✓ 全部候选已判完（可关闭窗口或按 Q）")
        # 必须放在 self.index 前移**之后**：列表高亮用的是 self.index，
        # 提前刷新会让光标停在刚判过的那一行，而右侧画面已经跳走了。
        self._refresh_listbox()

    def undo(self):
        last = pop_last_decision(self.decisions_path)
        if not last:
            self.status.configure(text="没有可撤销的决策")
            return
        for cand in self.candidates:
            if cand.episode_id == last["episode_id"]:
                cand.status = "pending"
                self.index = cand.index
                break
        self.load_current()
        self._refresh_listbox()
        self.status.configure(text=f"已撤销：{last['episode_id']}")

    # ── 渲染 ────────────────────────────────────────────────────────────
    def _render_episode(self):
        cand = self._current()
        if cand is None:
            self.canvas.draw_idle()
            return
        has_track = cand.dist is not None
        self._set_curve_visibility(has_track)
        if has_track:
            steps = np.arange(cand.dist.shape[0])
            self.dist_line.set_data(steps, cand.dist)
            self.ax_curve.set_xlim(0, max(1, steps[-1]))
            self.ax_curve.set_ylim(0, max(float(np.nanmax(cand.dist)) * 1.1, IN_BOX_R * 1.2))
            if cand.z is not None:
                self.z_line.set_data(steps, cand.z)
                self.lift_line.set_ydata([cand.init_z + MIN_BOX_LIFT] * 2)
                self.ax_z.set_ylim(float(np.nanmin(cand.z)) - 0.01,
                                   max(float(np.nanmax(cand.z)), cand.init_z) + 0.02)
            if cand.jaw is not None:
                self.jaw_line.set_data(steps, cand.jaw * 0.1)
            handles = [self.dist_line, self.in_box_line] + \
                      ([self.z_line, self.lift_line] if cand.z is not None else [])
            self.ax_curve.legend(handles, [h.get_label() for h in handles],
                                 loc="upper right", fontsize=7,
                                 facecolor=SIDE_BG, labelcolor=TITLE_COLOR)

        if self.im_artist is not None:
            self.im_artist.remove()
            self.im_artist = None
        if self.text_artist is not None:
            self.text_artist.remove()
            self.text_artist = None

        first = cand.raw_frame(0)
        if first is not None:
            self.im_artist = self.ax_img.imshow(
                _normalize_image(first, self.args.flip, self.args.flip))
        elif cand.frames_pixels:
            strip = np.concatenate([img for _s, img in cand.frames_pixels], axis=1)
            self.im_artist = self.ax_img.imshow(strip)
            self.ax_img.set_title(
                f"{cand.episode_id}   关键帧: {' | '.join(s for s, _ in cand.frames)}"
                f"（无 raw npz，不能回放）", color=TITLE_COLOR, fontsize=9)
        else:
            self.text_artist = self.ax_img.text(
                0.5, 0.5, f"无图像\n{cand.error or '未提供 --raw-dir / --frames-dir'}",
                color="#f88", ha="center", va="center", transform=self.ax_img.transAxes)
            self.ax_img.set_title(cand.episode_id, color=TITLE_COLOR, fontsize=10)

        if first is not None:
            self.ax_img.set_title(f"{cand.episode_id}   step 0/{cand.steps - 1}",
                                  color=TITLE_COLOR, fontsize=10)
        self._refresh_cursor()
        self.canvas.draw_idle()
        self._update_info(cand)
        self._sync_scale()

    def _refresh_cursor(self):
        cand = self._current()
        if cand is None:
            self.frame_values.configure(text="")
            return
        self.cursor.set_xdata([self.frame, self.frame])

        def value_text(values):
            if values is None or not 0 <= self.frame < len(values):
                return "n/a"
            value = float(values[self.frame])
            if not np.isfinite(value):
                return "n/a"
            return f"{value:.6f} m ({value * 1000:.3f} mm)"

        self.frame_values.configure(
            text=f"当前帧 {self.frame}   红块 XY 离盒心: {value_text(cand.dist)}"
                 f"   红块中心 Z: {value_text(cand.z)}")

    def _refresh_frame(self):
        """翻帧：只换图像数组 + 游标（不重建 axes）。"""
        cand = self._current()
        if cand is None:
            return
        raw = cand.raw_frame(self.frame)
        if raw is None:
            return                      # 关键帧模式是静态图
        if self.im_artist is not None:
            self.im_artist.set_data(_normalize_image(raw, self.args.flip, self.args.flip))
            self.ax_img.set_title(f"{cand.episode_id}   step {self.frame}/{cand.steps - 1}",
                                  color=TITLE_COLOR, fontsize=10)
        self._refresh_cursor()
        self.canvas.draw_idle()
        self._sync_scale()

    def _update_info(self, cand: Candidate):
        auto = cand.auto
        lines = [
            f"id      : {cand.episode_id}",
            f"状态    : {cand.status}",
            f"判据    : {auto.get('verdict', '?')}",
            f"d_end   : {_fmt(auto.get('dxy_end'))} m  (≤{IN_BOX_R})",
            f"dz      : {_fmt(auto.get('delta_red_z'))} m  (≥{MIN_BOX_LIFT})",
            f"released: {auto.get('released')}   xy_src: {auto.get('xy_source')}",
            f"steps   : {cand.steps}",
        ]
        lines.append(f"raw     : {cand.raw_path.name}" if cand.raw_path
                     else ("raw     : 缺失（仅关键帧）" if cand.frames_pixels else "raw     : 缺失"))
        if cand.error:
            lines.append(f"⚠ {cand.error}")
        self.info.configure(text="\n".join(lines))
        approved = sum(1 for c in self.candidates if c.status == "approved")
        rejected = sum(1 for c in self.candidates if c.status == "rejected")
        skipped = sum(1 for c in self.candidates if c.status == "skip")
        self.progress.configure(
            text=f"第 {self.index + 1}/{len(self.candidates)} 集    "
                 f"批准 {approved}  拒绝 {rejected}  跳过 {skipped}  "
                 f"待判 {len(self.candidates) - approved - rejected - skipped}")

    def _refresh_listbox(self):
        self.listbox.delete(0, "end")
        mark = {"approved": "✅", "rejected": "❌", "skip": "⏭", "pending": "  "}
        for cand in self.candidates:
            auto = cand.auto
            self.listbox.insert(
                "end",
                f"{mark.get(cand.status, '  ')} {cand.episode_id[-20:]}  "
                f"d={_fmt(auto.get('dxy_end'), 3)} z={_fmt(auto.get('delta_red_z'), 3)}")
        if self.candidates:
            self.listbox.see(self.index)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(self.index)

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel and int(sel[0]) != self.index:
            self.index = int(sel[0])
            self.load_current()

    # ── 播放 ────────────────────────────────────────────────────────────
    def toggle_play(self):
        self.playing = not self.playing
        if self._play_after_id is not None:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None
        if self.playing:
            self._tick()

    def _tick(self):
        self._play_after_id = None
        if not self.playing:
            return
        cand = self._current()
        if cand is None:
            return
        self.frame = (self.frame + 1) % max(1, cand.steps)
        self._refresh_frame()
        self._play_after_id = self.root.after(
            max(10, int(1000 / max(1.0, self.fps))), self._tick)

    def step_frame(self, delta: int):
        cand = self._current()
        if cand is None:
            return
        self.frame = max(0, min(self.frame + delta, cand.steps - 1))
        self._refresh_frame()

    # ── 进度条（可拖动定位）─────────────────────────────────────────────
    def _on_scale(self, value):
        """滑块被拖动 → 定位到该帧。

        播放中拖动由 press/release 负责暂停与恢复；程序回写滑块时
        `_syncing_scale` 为真，直接返回，否则会与 `_sync_scale` 互相触发。
        """
        if self._syncing_scale:
            return
        cand = self._current()
        if cand is None:
            return
        target = max(0, min(int(float(value)), max(0, cand.steps - 1)))
        if target == self.frame:
            return                      # 拖动过程中大量重复值，跳过无谓重绘
        self.frame = target
        self._refresh_frame()

    def _on_scale_press(self, event):
        """进度条任意位置均可按住拖动；暂停计时，防止播放抢帧。"""
        if str(self.scale.cget("state")) == "disabled":
            return "break"
        self._scrubbing = True
        self._resume_after_scrub = self.playing
        if self.playing:
            self.toggle_play()
        self._seek_scale_at(event.x)
        return "break"

    def _seek_scale_at(self, x):
        lo, hi = float(self.scale.cget("from")), float(self.scale.cget("to"))
        left, right = self.scale.coords(lo)[0], self.scale.coords(hi)[0]
        frac = min(1.0, max(0.0, (x - left) / max(1, right - left)))
        self.scale.set(round(lo + frac * (hi - lo)))
        self._on_scale(self.scale.get())

    def _on_scale_motion(self, event):
        if self._scrubbing:
            self._seek_scale_at(event.x)
        return "break"

    def _on_scale_release(self, _event):
        self._scrubbing = False
        if self._resume_after_scrub:
            self._resume_after_scrub = False
            self.toggle_play()
        return "break"

    def _scale_step(self, delta: int):
        """滑块上的 ←/→/Home/End：转给 step_frame 并吃掉事件。

        控件级绑定先于 Scale 的类绑定执行，返回 "break" 可阻止类绑定
        也去改滑块值（否则一次按键会既翻帧又挪滑块）。
        """
        self.step_frame(delta)
        return "break"

    def _sync_scale(self):
        """把滑块的范围与位置对齐到当前集；没有 raw（只看静态关键帧）就禁用拖动。"""
        cand = self._current()
        scrubbable = cand is not None and cand.raw_frame(0) is not None
        steps = max(1, cand.steps) if cand is not None else 1
        self._syncing_scale = True
        try:
            self.scale.configure(to=max(0, steps - 1),
                                 state=("normal" if scrubbable else "disabled"))
            self.scale.set(self.frame)
        finally:
            self._syncing_scale = False
        self.frame_label.configure(
            text=(f"{self.frame}/{steps - 1} ({100 * self.frame / max(1, steps - 1):.0f}%)"
                  if scrubbable else "无 raw，不可拖动"))

    def _bind_keys(self):
        bind = self.root.bind_all
        bind("<space>", lambda e: self.toggle_play())
        bind("<Left>", lambda e: self.step_frame(-1))
        bind("<Right>", lambda e: self.step_frame(1))
        bind("<Home>", lambda e: self.step_frame(-10 ** 9))
        bind("<End>", lambda e: self.step_frame(10 ** 9))
        bind("<KeyPress-a>", lambda e: self.judge("approved"))
        bind("<KeyPress-r>", lambda e: self.judge("rejected"))
        bind("<KeyPress-s>", lambda e: self.judge("skip"))
        bind("<KeyPress-u>", lambda e: self.undo())
        bind("<KeyPress-p>", lambda e: self.step_case(-1))
        bind("<KeyPress-n>", lambda e: self.step_case(1))
        bind("<KeyPress-q>", lambda e: self.root.destroy())
        bind("<Escape>", lambda e: self.root.destroy())
        # 滑块若拿到焦点，Scale 的类绑定会自己消费方向键/Home/End，这里在控件级先接管
        self.scale.bind("<Left>", lambda e: self._scale_step(-1))
        self.scale.bind("<Right>", lambda e: self._scale_step(1))
        self.scale.bind("<Home>", lambda e: self._scale_step(-10 ** 9))
        self.scale.bind("<End>", lambda e: self._scale_step(10 ** 9))

    def run(self):
        self.root.mainloop()


# ── 入口 ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="collect_rollouts.py 产出的 manifest.jsonl")
    p.add_argument("--decisions", help="决策 JSONL（追加写入；--check 时可省略）")
    p.add_argument("--reviewer", default=os.environ.get("USERNAME") or "reviewer")
    p.add_argument("--raw-dir", help="raw rollout npz 目录（可回放整段 + 画红块轨迹）")
    p.add_argument("--frames-dir", help="下载的关键帧目录（<id>_final.png / <id>_min.png）")
    p.add_argument("--only", choices=["pending", "all"], default="pending",
                   help="pending=只审自动成功候选（默认）；all=连失败集一起看")
    p.add_argument("--flip", action="store_true",
                   help="上下+左右翻转显示（DatasetsCollector 示教格式用）；"
                        "RECAP rollout 是正向的，默认不翻转")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--geometry", default="1500x900")
    p.add_argument("--check", action="store_true",
                   help="不开窗：只打印候选队列与素材是否齐全，便于脚本化校验")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_file():
        print(f"manifest 不存在: {manifest}", file=sys.stderr)
        return 2

    rows = select_rows(manifest, args.only)
    if not rows:
        print(f"没有符合条件的 episode（--only {args.only}）")
        return 0
    decisions = load_decisions(Path(args.decisions).expanduser()) if args.decisions else {}
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None
    frames_dir = Path(args.frames_dir).expanduser() if args.frames_dir else None

    candidates = []
    for i, row in enumerate(rows):
        eid = str(row.get("episode_id"))
        cand = Candidate(index=i, row=row,
                         status=(decisions.get(eid) or {}).get("decision", "pending"))
        candidates.append(cand)

    if args.check:
        print(f"{'episode_id':<30} {'状态':<8} {'d_end':>8} {'dz':>9}  素材")
        n_raw = n_frames = n_empty = 0
        for cand in candidates:
            build_candidate(cand, raw_dir, frames_dir, args.flip, args.flip)
            if cand.raw_path:
                src = "raw"
                n_raw += 1
            elif cand.frames_pixels:
                src = "关键帧"
                n_frames += 1
            else:
                src = "缺失"
                n_empty += 1
            auto = cand.auto
            print(f"{cand.episode_id:<30} {cand.status:<8} "
                  f"{_fmt(auto.get('dxy_end'), 4):>8} {_fmt(auto.get('delta_red_z'), 4):>9}  {src}"
                  + (f"   ⚠ {cand.error}" if cand.error else ""))
        print(f"\n共 {len(candidates)} 个候选：raw {n_raw}，关键帧 {n_frames}，无素材 {n_empty}")
        if args.decisions:
            print(f"已判 {sum(1 for c in candidates if c.status != 'pending')}，"
                  f"决策文件 {Path(args.decisions).expanduser()}")
        return 0

    if not args.decisions:
        print("--decisions 是审阅模式的必填参数（--check 才可省略）", file=sys.stderr)
        return 2
    if not raw_dir and not frames_dir:
        print("警告：既没给 --raw-dir 也没给 --frames-dir，窗口里将没有画面。", file=sys.stderr)

    ReviewApp(args, candidates, decisions).run()
    done = sum(1 for c in candidates if c.status != "pending")
    print(f"审阅结束：已判 {done}/{len(candidates)}，决策写入 {args.decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
