# DatasetsCollector — LIBERO 数据采集与可视化

本项目用于 LIBERO 双臂 block-grasping 任务的遥操作数据采集、格式转换和可视化分析。

## 项目结构

```
DatasetsCollector\
    collect_datasets.py         # Joy-Con 遥操作采集
    env_client.py               # Unity TCP 客户端
    convert_to_rlds.py          # .npz → 训练格式 + norm stats
    filter_idle_frames.py       # 发呆帧过滤
    analyze_bin_distribution.py # Bin 分布分析
    npz_visualizer.py           # 数据可视化 GUI (RoboView)
    demos\                      # 采集的 .npz 输出
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

## 3. 发呆帧过滤

遥操作数据中大部分帧是操作者微调/犹豫的姿态（位移接近零），会导致模型学到退化策略"永远不动"。过滤掉这些帧可以大幅提升训练质量。

```bash
# 推荐：过滤约 30% 发呆帧
python filter_idle_frames.py -i ./demos -o ./demos_filtered --pos_threshold 0.0015 --rot_threshold 0.005

# 从 network 跨项目调用
python filter_idle_frames.py -i ../datasets/trajectories -o ../datasets/trajectories_filtered
```

### 3.1 阈值说明

一帧只要**任意一条手臂**满足以下任一条件就会被保留：

| 条件 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 位移幅度 > 阈值 | `--pos_threshold` | 0.0015 | EEF 位移量 (m)，单帧均值 ~0.001，推 0.001-0.002 |
| 旋转幅度 > 阈值 | `--rot_threshold` | 0.005 | EEF 旋转量 (rad)，单帧均值 ~0.003，推 0.003-0.01 |
| 夹爪变化 > 阈值 | `--grip_threshold` | 0.01 | 夹爪值变化 (0=开 1=关)，推 0.01（检测开关切换） |

**调参指南**：

| 场景 | 参数 | 效果 |
|------|------|------|
| 裁得更狠（保留更少） | `--pos_threshold 0.002 --rot_threshold 0.01` | 只保留明显移动帧 |
| 裁得更松（保留更多） | `--pos_threshold 0.0005 --rot_threshold 0.003` | 保留微动帧 |
| 只看右臂 | 调高 rot_threshold + 关闭夹爪检测 | 适合单手操作数据 |

### 3.2 效果验证

用 `analyze_bin_distribution.py` 对比过滤前后的 bin 分布：

```bash
# 过滤前
python analyze_bin_distribution.py -i ./demos -o ./bin_orig

# 过滤后
python analyze_bin_distribution.py -i ./demos_filtered -o ./bin_filtered

# 过滤后 + 5 帧累积
python analyze_bin_distribution.py -i ./demos_filtered -o ./bin_step5 --step_skip 5
```

Verdict 解读：
- `EXCELLENT` (<10%) — 分布均匀，无模式坍缩
- `GOOD` (10-20%) — 轻微峰值，可接受
- `OK` (20-35%) — 中等峰值，建议加 step_skip
- `WEAK` (35-50%) — 明显峰值，需要过滤 + step_skip
- `BAD (collapse)` (>50%) — 严重模式坍缩，数据不可用

## 4. 数据可视化 (RoboView)

### 4.1 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

软件界面使用 Python 自带的 `tkinter`。如果你的 Python 发行版没有包含 `tkinter`，需要安装带 Tk 支持的 Python。

### 4.2 启动软件

```bash
python npz_visualizer.py
```

### 4.3 导入数据

- **导入单个 NPZ**：点击”导入 NPZ”，选择一个 `.npz` 文件
- **导入文件夹**：点击”导入文件夹”，选择 `demos/` 目录

### 4.4 播放和分析

- 点击 `Play` / `Pause` 或按空格键播放/暂停
- 拖动 `step` 滑块跳转帧
- 左右方向键逐帧前进/后退
- 拖动 `fps` 滑块调整播放速度
- 窗口右侧显示 action、关节、末端位姿和夹爪曲线

### 4.5 命令行模式

```bash
python npz_visualizer.py demos --list                    # 列出所有 episode
python npz_visualizer.py demos --episode-index 1          # 打开第 1 条
python npz_visualizer.py episode.npz                      # 直接打开文件
python npz_visualizer.py demos --concat-all               # 合并所有 episode 播放
```

### 4.6 支持的数据字段

`agentview_image`, `action`, `robot0/1_joint_pos`, `robot0/1_eef_pos`, `robot0/1_eef_quat`, `robot0/1_gripper_qpos`, `language_instruction`, `success`
