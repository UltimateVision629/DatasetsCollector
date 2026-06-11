# RoboView NPZ Visualizer

RoboView 是一个用于查看 Unity 机械臂采集数据的 `.npz` 可视化软件。它可以先导入 `.npz` 文件，再对单条 episode 做图像回放、动作曲线、关节状态和末端位姿分析。

## 1. 环境准备

进入项目目录：

```bash
cd D:\Work\roboview
```

安装依赖：

```bash
pip install -r requirements.txt
```

依赖只有：

- `numpy`
- `matplotlib`

软件界面使用 Python 自带的 `tkinter`。如果你的 Python 发行版没有包含 `tkinter`，需要安装带 Tk 支持的 Python。

## 2. 启动软件

直接运行：

```bash
python npz_visualizer.py
```

启动后会打开 RoboView 的导入窗口。这个窗口用于选择数据文件，并不是最终分析窗口。

## 3. 导入数据

### 导入单个 NPZ

点击窗口顶部的“导入 NPZ”，选择一个 `.npz` 文件。

适合场景：

- 你只想分析某一条采集轨迹。
- 文件不在当前项目的 `demos/` 目录里。

### 导入文件夹

点击“导入文件夹”，选择一个包含 `.npz` 的目录，例如：

```text
D:\Work\roboview\demos
```

软件会列出该目录下所有 `.npz` 文件。你可以在列表中逐条选择，然后点击“开始分析”。

注意：导入文件夹后默认仍然是一条一条分析，不会自动把多条 episode 合并播放。

## 4. 查看基础信息

导入文件后，主窗口下方会显示当前选中 episode 的基础信息：

- 文件路径
- 任务名
- 是否成功
- 步数
- 图像字段名
- action 字段名
- 所有数组的 `shape` 和 `dtype`

例如你的当前数据中，单条轨迹通常类似：

```text
agentview_image  shape=(347, 224, 224, 3)  dtype=uint8
action           shape=(347, 14)           dtype=float32
```

这表示源数据本身就是 347 帧、224x224 分辨率。可视化脚本不会对源数据降采样、裁剪或压缩。

## 5. 开始分析

在列表中选中一条 `.npz` 后，点击“开始分析”。

软件会打开一个新的分析窗口，包含：

- 左上：相机画面回放
- 左下：当前帧实时数值
- 右侧：动作、关节、末端位姿和夹爪曲线
- 底部：帧滑块和播放速度滑块

## 6. 播放和交互

分析窗口支持以下操作：

- 点击 `Play` 开始播放。
- 再次点击 `Pause` 暂停播放。
- 拖动 `step` 滑块跳转到任意帧。
- 拖动 `fps` 滑块实时调整播放速度。
- 按空格键播放或暂停。
- 按左右方向键逐帧前进或后退。
- 按 `Home` 跳到第一帧。
- 按 `End` 跳到最后一帧。

播放速度说明：

- `fps=1` 表示每秒播放 1 帧。
- `fps=30` 表示每秒播放 30 帧。
- `fps=120` 表示快速浏览。

播放速度只影响可视化回放，不会修改 `.npz` 文件。

## 7. 图像方向

Unity 相机帧有时会以上下颠倒的方式保存到数组中。RoboView 默认会在显示时做一次垂直翻转，让画面看起来是正的。

在导入窗口中可以通过“垂直翻转图像”复选框控制：

- 勾选：显示时翻转图像，适合当前 Unity 数据。
- 取消：按 `.npz` 原始数组方向显示。

这个操作只影响显示，不会改写源数据。

## 8. 数据曲线含义

### Action 曲线

采集脚本保存的是 14 维 action：

```text
[robot0: dx, dy, dz, dRx, dRy, dRz, gripper,
 robot1: dx, dy, dz, dRx, dRy, dRz, gripper]
```

分析窗口会拆成：

- `Action position deltas`
- `Action rotation deltas`
- `Gripper values`

### 机械臂状态

窗口还会显示：

- `Robot joint positions`：双臂关节角
- `End-effector positions`：双臂末端位置
- `End-effector quaternions`：双臂末端姿态四元数
- `Gripper values`：夹爪状态

曲线上的黑色虚线会跟随当前播放帧同步移动。

## 9. 命令行模式

除了软件界面，也可以直接用命令行。

列出 `demos/` 下所有 episode：

```bash
python npz_visualizer.py demos --list
```

只查看第 1 条基础信息：

```bash
python npz_visualizer.py demos --episode-index 1 --info-only
```

直接打开第 1 条分析窗口：

```bash
python npz_visualizer.py demos --episode-index 1
```

直接打开某个文件：

```bash
python npz_visualizer.py path\to\episode.npz
```

指定初始播放速度：

```bash
python npz_visualizer.py demos --episode-index 1 --fps 60
```

关闭默认图像垂直翻转：

```bash
python npz_visualizer.py demos --episode-index 1 --no-flip-image
```

临时把目录下所有 episode 合并成一条时间线查看：

```bash
python npz_visualizer.py demos --concat-all
```

## 10. 支持的数据字段

脚本优先识别以下字段：

- `agentview_image`
- `action`
- `robot0_joint_pos`
- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- `robot1_joint_pos`
- `robot1_eef_pos`
- `robot1_eef_quat`
- `robot1_gripper_qpos`
- `language_instruction`
- `success`

如果图像字段名不是 `agentview_image`，脚本会尝试从包含 `image`、`rgb`、`camera`、`agentview` 的 key 中自动推断。

## 11. 常见问题

### 为什么一条数据只有 300 多帧？

因为每个 `.npz` 是一次单独采集的 episode。你的当前源数据里，单条 episode 本身就是 230 到 399 帧左右。这不是可视化脚本造成的。

### 为什么分辨率只有 224x224？

因为 `.npz` 中保存的 `agentview_image` 原始数组就是 `N x 224 x 224 x 3`。脚本不会降低分辨率。

### 为什么画面会上下颠倒？

这是 Unity 图像坐标和显示坐标方向不同导致的常见现象。默认勾选“垂直翻转图像”即可正常观看。

### 导入文件夹后会不会自动合并？

不会。软件界面中导入文件夹只是为了方便选择文件，分析仍然是一条一条进行。只有命令行显式使用 `--concat-all` 时才会合并。
