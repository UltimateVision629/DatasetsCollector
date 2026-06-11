# DatasetsCollector — LIBERO 数据采集与可视化

本项目用于 LIBERO 双臂 block-grasping 任务的遥操作数据采集、格式转换和可视化分析。

## 项目结构

```
DatasetsCollector\
    collect_datasets.py     # Joy-Con 遥操作采集
    env_client.py           # Unity TCP 客户端
    convert_to_rlds.py      # .npz → 训练格式 + norm stats
    npz_visualizer.py       # 数据可视化 GUI (RoboView)
    demos\                  # 采集的 .npz 输出
```

## 1. 数据采集

通过 Joy-Con 遥控 Unity 中的 SO100 双臂，录制 (observation, action) 轨迹。

```bash
# 1. 先启动 Unity Play 模式
# 2. 运行采集脚本
python collect_datasets.py
```

控制：
- 右手柄 → 右臂 (robot_0)
- 左手柄 → 左臂 (robot_1)
- ZL/ZR → 夹爪切换
- A 键 → 保存成功 episode
- Y 键 → 丢弃并复位机械臂

输出：`demos/episode_XXXX_<task>_<timestamp>_success.npz`

## 2. 格式转换

将采集的 `.npz` 转为 VLA-Adapter 训练格式，并计算归一化统计量：

```bash
python convert_to_rlds.py --input_dir ./demos --output_dir ../network/dataset
```

## 3. 数据可视化 (RoboView)

### 3.1 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

软件界面使用 Python 自带的 `tkinter`。如果你的 Python 发行版没有包含 `tkinter`，需要安装带 Tk 支持的 Python。

### 3.2 启动软件

```bash
python npz_visualizer.py
```

### 3.3 导入数据

- **导入单个 NPZ**：点击”导入 NPZ”，选择一个 `.npz` 文件
- **导入文件夹**：点击”导入文件夹”，选择 `demos/` 目录

### 3.4 播放和分析

- 点击 `Play` / `Pause` 或按空格键播放/暂停
- 拖动 `step` 滑块跳转帧
- 左右方向键逐帧前进/后退
- 拖动 `fps` 滑块调整播放速度
- 窗口右侧显示 action、关节、末端位姿和夹爪曲线

### 3.5 命令行模式

```bash
python npz_visualizer.py demos --list                    # 列出所有 episode
python npz_visualizer.py demos --episode-index 1          # 打开第 1 条
python npz_visualizer.py episode.npz                      # 直接打开文件
python npz_visualizer.py demos --concat-all               # 合并所有 episode 播放
```

### 3.6 支持的数据字段

`agentview_image`, `action`, `robot0/1_joint_pos`, `robot0/1_eef_pos`, `robot0/1_eef_quat`, `robot0/1_gripper_qpos`, `language_instruction`, `success`
