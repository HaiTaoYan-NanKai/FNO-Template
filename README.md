
# FNO Template（固体力学参数反演）

本项目基于 **Fourier Neural Operator (FNO-1D)**，实现从载荷-位移曲线反演材料参数（屈服强度 σ<sub>Y</sub>、硬化指数 n）。结构参考 PINN 项目：配置/日志、数据预处理、模型与训练、推理与可视化分层清晰，便于维护与后续扩展。

## 目录结构
```
Allocation/        # 配置与日志工具
AnalyzeVisualize/  # 可视化/中文字体设置
DataPrep/          # Excel读取、清洗、标准化、DataLoader
NNNArchitecture/   # FNO 模型、训练、推理
Data.xlsx          # 实验模拟数据（与根目录同层）
```

## 数据格式（与 Excel 一致）
- 第1列：屈服强度（float）
- 第2列：硬化指数（float）
- 第3–103列：加载阶段载荷值（等间距位移采样下的 **Y 轴**）
- 第104–204列：卸载阶段载荷值  
共 121 行样本（可增补）。(默认 80% 训练；训练内部切 10% 做验证；20% 测试)

> 说明：我们只记录 **载荷值**（Y），默认所有样本的 X 轴位移采样点相同且等间距。

## 快速开始
```bash
pip install -r requirements.txt

# 训练（自动选择 GPU/CPU；支持断点续训）
python -m NNNArchitecture.train

# 推理（可指定新的 Excel）
python -m NNNArchitecture.inference --excel_path Data.xlsx
```

训练过程会在：
- `./checkpoints/` 保存 `best_model.pth`、周期性检查点与 `scalers.json`
- `./logs/` 写入 `loss.csv`、`train.log`、`test_metrics.json`
- `./plots/` 保存散点与曲线对比图（曲线对比使用**简化前向**示意，可换成真实前向模型）

## 关键设计
- **标准化/反变换**：输入曲线与参数各自标准化；推理阶段自动反变换回真实物理量。
- **健壮性**：Excel 读取/列数校验/缺失值删除/分位裁剪（0.5%–99.5%）等。
- **训练**：`ReduceLROnPlateau`、早停、梯度裁剪、最优模型保存、断点续训。
- **可视化**：预测 vs 真实散点；（可选）用 `approx_forward_curve` 生成简化前向曲线做示意对比。
- **可配置**：`Allocation/config.json` 管理所有超参数与路径。

## 常见问题
1. **中文字体乱码**：安装 `Noto Sans CJK SC` 或在 `visualisation.py` 的候选列表里添加本机字体。
2. **Windows 卡在 DataLoader**：`num_workers` 设为 0（配置文件已默认）。
3. **Excel 打不开/被占用**：关闭占用进程，或另存一份再读；确保安装 `openpyxl`。

## 致谢
- 需求与数据规范来自你的任务书与先前单文件脚本（在本模版中已模块化实现）。
