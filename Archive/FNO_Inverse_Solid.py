#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO-1D 用于固体力学参数反演（屈服强度 σ_Y、硬化指数 n）
================================================================

- 数据来源：Excel 文件 "Data.xlsx"（与本脚本位于同一文件夹）
- 数据格式：
  第1列：屈服强度（float）
  第2列：硬化指数（float）
  第3-103列：加载阶段的载荷值（等间距位移采样下的 Y 轴）
  第104-204列：卸载阶段的载荷值（等间距位移采样下的 Y 轴）
  共 204 列；共有 121 行样本（默认设置）。

- 任务：使用 1D FNO（Fourier Neural Operator）从曲线（输入）反演材料参数（输出）。
- 训练/验证/测试：默认 80% 训练 + 20% 测试；并从训练集中切出 10% 作为验证。
  因为需要“基于验证集最低 MSE 保存最佳模型”，所以实际为：
      训练 72%、验证 8%、测试 20%。
  若数据量很小（本例 121 条），该比例在偏小数据集上较为稳健（代码内可通过参数修改）。

- 关键特性：
  * 数据加载与清洗（缺失值、异常值处理与维度一致性校验）
  * 标准化归一化（输入/输出），含反变换逻辑，训练后保存 scaler 以便推理时还原物理量
  * FNO-1D 网络（可配置层数、模数、通道数/宽度等超参数）
  * 可复现实验（随机种子固定）
  * 最佳模型保存（验证集最优 MSE），并支持断点续训与每 100 epoch 保存检查点
  * 日志记录与可视化（loss.csv；预测-真实散点图；（可选）曲线预测可视化示例）
  * 健壮性与错误处理（Excel 读取失败、数据格式不符、形状不一致等）
  * 支持 CPU/GPU 训练（自动选择设备）

依赖：
  - Python 3.9+
  - torch 1.10+
  - pandas, numpy, matplotlib
  - openpyxl（读取 .xlsx 时通常需要）

使用：
  1) 安装依赖：
     pip install torch pandas numpy matplotlib openpyxl
  2) 将 Data.xlsx 与本脚本置于同一文件夹。
  3) 运行：
     python fno_inverse_solid.py --epochs 1000 --batch_size 16
     （更多命令行参数见下文 argparse 部分）

备注：
  - “载荷-位移曲线的预测与真实曲线可视化”在仅进行参数反演的设定下并不严格可行，
    因为模型输出的是材料参数而非整条曲线。为了满足可视化需求，本脚本提供了**可选的**简化前向模型
    `approx_forward_curve(sigma_y, n, length)` 以基于预测参数生成一条“近似曲线”，仅用于示意。
    如果您有真实的物理前向模型，请将 `approx_forward_curve` 替换为您的前向求解器以获得物理上更准确的对比。
"""

import argparse
import os
import sys
import math
import json
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ------------------------------
# 实用工具：随机种子、设备、日志目录
# ------------------------------

def set_seed(seed: int = 42) -> None:
    """固定随机种子以确保可复现性。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """选择设备（优先 GPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ------------------------------
# 简单标准化工具（避免外部额外依赖）
# ------------------------------

class StandardScaler1D:
    """一维特征标准化（可批量），存储 mean/std 并支持反变换。"""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> 'StandardScaler1D':
        """
        x: 形状 (N, F) 或 (N,)；对每个特征/列做标准化。
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        self.std_ = np.std(x, axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler1D 未拟合，请先调用 fit().")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler1D 未拟合，请先调用 fit().")
        return x * self.std_ + self.mean_

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist()
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'StandardScaler1D':
        obj = StandardScaler1D()
        if d["mean"] is not None:
            obj.mean_ = np.asarray(d["mean"], dtype=np.float32)
        if d["std"] is not None:
            obj.std_ = np.asarray(d["std"], dtype=np.float32)
        return obj


# ------------------------------
# 数据集：Excel 加载、清洗与校验
# ------------------------------

@dataclass
class DataConfig:
    excel_path: str = "Data.xlsx"
    expected_cols: int = 204   # 1:σ_Y, 2:n, 3..204: 曲线，共 202 点
    start_curve_col: int = 3
    end_curve_col: int = 204
    train_ratio: float = 0.80   # 总体 80% 训练（再切出 10% 验证）
    val_within_train_ratio: float = 0.10  # 训练中的 10% 做验证
    random_seed: int = 42
    clip_outliers: bool = True
    clip_quantiles: Tuple[float, float] = (0.005, 0.995)  # 0.5%~99.5% 分位裁剪
    dropna: bool = True


class ExcelCurveDataset(Dataset):
    """
    从 Excel 读取数据的 PyTorch Dataset。
    - 自动完成缺失值处理、异常值裁剪、形状与列数校验。
    - 标准化：由外部提供 scaler（fit 在整个训练+验证集上）。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_cfg: DataConfig,
        input_scaler: Optional[StandardScaler1D] = None,
        output_scaler: Optional[StandardScaler1D] = None,
    ):
        self.data_cfg = data_cfg

        # 1) 只取前 expected_cols 列；若不足则报错
        if df.shape[1] < data_cfg.expected_cols:
            raise ValueError(f"Excel 列数不足：期望至少 {data_cfg.expected_cols} 列，实际 {df.shape[1]} 列。")
        df = df.iloc[:, :data_cfg.expected_cols].copy()

        # 2) 列命名（便于调试）
        new_cols = ["sigma_y", "n"] + [f"curve_{i}" for i in range(1, data_cfg.end_curve_col - data_cfg.start_curve_col + 2)]
        df.columns = new_cols

        # 3) 缺失值处理
        if data_cfg.dropna:
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                print(f"[数据清洗] 已删除含缺失值的样本：{before - after} 条")

        # 4) 异常值裁剪（对每个曲线列分位数裁剪）
        if data_cfg.clip_outliers:
            curve_cols = [c for c in df.columns if c.startswith("curve_")]
            q_low = df[curve_cols].quantile(data_cfg.clip_quantiles[0])
            q_high = df[curve_cols].quantile(data_cfg.clip_quantiles[1])
            df[curve_cols] = df[curve_cols].clip(lower=q_low, upper=q_high, axis=1)

        # 5) 基本形状校验
        num_curve_cols = data_cfg.end_curve_col - data_cfg.start_curve_col + 1
        if len([c for c in df.columns if c.startswith("curve_")]) != num_curve_cols:
            raise ValueError(f"曲线列数不正确，应为 {num_curve_cols}，实际为 {len([c for c in df.columns if c.startswith('curve_')])}。")

        self.df = df.reset_index(drop=True)

        # 6) 分离输入输出
        y = self.df[["sigma_y", "n"]].values.astype(np.float32)   # (N, 2)
        x = self.df[[c for c in self.df.columns if c.startswith("curve_")]].values.astype(np.float32)  # (N, 202)

        # 7) 标准化（若提供 scaler 则使用，否则原样）
        if input_scaler is not None:
            x = input_scaler.transform(x)
        if output_scaler is not None:
            y = output_scaler.transform(y)

        # 存储
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回形状：输入 (F,)；输出 (2,)
        x = torch.from_numpy(self.x[idx])  # (202,)
        y = torch.from_numpy(self.y[idx])  # (2,)
        return x, y


def load_excel_dataframe(path: str) -> pd.DataFrame:
    """读取 Excel；提供友好错误信息。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel 文件未找到：{path}（请确认 Data.xlsx 与脚本在同一目录）")
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"读取 Excel 失败：{e}\n请确认已安装 openpyxl，并且文件未被占用。")
    if df.shape[0] < 5 or df.shape[1] < 10:
        print(f"[警告] Excel 表格数据量较小或列数异常：shape={df.shape}，请确认格式是否正确。")
    return df


def setup_chinese_font():
    # 常见可用字体：按系统优先级尝试
    candidates = [
        "Noto Sans CJK SC",      # Linux/通用（推荐）
        "Source Han Sans SC",    # 思源黑体
        "Microsoft YaHei",       # Windows 微软雅黑
        "SimHei",                # 黑体
        "PingFang SC",           # macOS 苹方
        "WenQuanYi Zen Hei",     # Linux 文泉驿
        "Songti SC", "STHeiti"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.sans-serif"] = [name]
            print(f"[Matplotlib] 使用中文字体：{name}")
            break
    else:
        print("[Matplotlib] 未找到常见中文字体，可能出现中文乱码；建议安装 Noto Sans CJK SC。")
    # 让负号正常显示（很多中文字体不含 Unicode minus）
    rcParams["axes.unicode_minus"] = False


# ------------------------------
# FNO-1D 网络
# ------------------------------

class SpectralConv1d(nn.Module):
    """
    基于 rFFT 的 1D 频域卷积层：只学习前若干低频模（modes），其余高频置零。
    参考 FNO 论文思想：通过 FFT -> 低频线性变换 -> iFFT，实现全局非局部建模能力。
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 仅使用的前若干频率模

        # 复权重参数：形状 [in_c, out_c, modes]，使用两个实参数表示复数（real, imag）
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, a: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor) -> torch.Tensor:
        """
        复数乘法：a: [B, in_c, K] 的复张量，b_*: [in_c, out_c, K]
        返回： [B, out_c, K] 的复张量
        """
        # a: complex64/128，b_real/b_imag: real
        # einsum: bco, bic -> boc
        real = torch.einsum("bik, iok -> bok", a.real, b_real) - torch.einsum("bik, iok -> bok", a.imag, b_imag)
        imag = torch.einsum("bik, iok -> bok", a.real, b_imag) + torch.einsum("bik, iok -> bok", a.imag, b_real)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, N]
        """
        B, C_in, N = x.shape
        # rFFT：输出频谱 [B, C_in, N//2+1]（实信号的半谱）
        x_ft = torch.fft.rfft(x, n=N)  # complex tensor

        # 只保留低频模
        K = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.complex64, device=x.device)
        out_ft[:, :, :K] = self.compl_mul1d(
            x_ft[:, :, :K], self.weight_real[:, :, :K], self.weight_imag[:, :, :K]
        )

        # iFFT 回到时域
        x_out = torch.fft.irfft(out_ft, n=N)  # [B, C_out, N]
        return x_out


class FNO1d(nn.Module):
    """
    FNO-1D 主体：Lift -> (SpectralConv + 1x1Conv)*L -> GlobalAvgPool -> MLP 输出参数
    - 输入：曲线序列 [B, N]
    - 输出：两个标量参数 [B, 2]
    """
    def __init__(
        self,
        seq_len: int,
        in_channels: int = 1,
        width: int = 64,
        modes: int = 16,
        layers: int = 4,
        mlp_hidden: int = 128,
        out_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.width = width
        self.modes = modes
        self.layers = layers

        # Lift：1 -> width 通道（使用 1x1 Conv）
        self.input_proj = nn.Conv1d(in_channels, width, kernel_size=1)

        # 多个谱卷积块 + 1x1 conv（等价于论文里的 W）
        self.spectral_layers = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pointwise_layers = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(layers)])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 读出：全局平均池化后 -> MLP -> 2 参
        self.readout = nn.Sequential(
            nn.Linear(width, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, out_dim),
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.shape[1]
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N] -> expand -> [B, 1, N]
        return: [B, 2]
        """
        if x.ndim != 2:
            raise ValueError(f"FNO1d 期望输入形状 [B, N]，实际 {x.shape}")
        B, N = x.shape
        if N != self.seq_len:
            raise ValueError(f"FNO1d 序列长度不匹配：初始化为 {self.seq_len}，实际输入 {N}")
        x = x.unsqueeze(1)  # [B, 1, N]

        # Lift
        x = self.input_proj(x)  # [B, width, N]

        # (Spectral + Pointwise)*L
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            y = spec(x)  # [B, width, N]
            y = y + pw(x)  # 残差
            x = self.act(y)
            x = self.dropout(x)

        # Global Average Pooling over N
        x = x.mean(dim=-1)  # [B, width]

        # Readout
        out = self.readout(x)  # [B, 2]
        return out


# ------------------------------
# 训练与评估流程
# ------------------------------

@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 50           # 早停容忍
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    plot_dir: str = "./plots"
    save_every: int = 100        # 每 100 epoch 保存一次检查点
    resume: bool = False
    num_workers: int = 0


@dataclass
class ModelConfig:
    width: int = 64
    modes: int = 16
    layers: int = 4
    mlp_hidden: int = 128
    dropout: float = 0.0


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    在真实物理量尺度上计算评估指标（MSE、MAE、R2）。
    pred, true: (N, 2)
    """
    assert pred.shape == true.shape
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    # R^2
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"MSE": mse, "MAE": mae, "R2": r2}


def approx_forward_curve(sigma_y: float, n: float, length: int = 202) -> np.ndarray:
    """
    【可选】基于预测参数生成近似“载荷-位移”曲线，仅用于可视化对比，
    非真实物理前向求解器，请按需替换为真实模型。

    简化模型思路：
    - 加载阶段（前半段）：假设线弹性至屈服强度 σ_y，然后进入硬化阶段 ~ (strain)^n。
    - 卸载阶段（后半段）：弹性卸载（同斜率）。

    注意：仅示意用，参数尺度未与真实单位严格校准。
    """
    # 归一位移 0..1，切成加载/卸载两段
    t = np.linspace(0, 1, length, dtype=np.float32)
    half = length // 2
    load = np.zeros_like(t)

    # 假设弹性模量 E（未知），这里取一个常数因子使曲线具有基本形状
    E = max(1.0, sigma_y / 0.01)  # 仅为形状参数；避免 E=0

    # 加载段
    for i in range(half):
        strain = t[i]
        stress_elastic = E * strain
        if stress_elastic < sigma_y:
            load[i] = stress_elastic
        else:
            # 简化硬化：σ = σ_y + k * (strain - σ_y/E)^n
            k = sigma_y  # 仅作尺度
            load[i] = sigma_y + k * max(0.0, strain - sigma_y / E) ** max(n, 0.01)

    # 卸载段：从末点线性回弹到 0
    end_val = load[half - 1]
    for i in range(half, length):
        alpha = (i - half) / (length - half - 1 + 1e-8)
        load[i] = (1 - alpha) * end_val  # 线性回退

    return load


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        input_scaler: StandardScaler1D,
        output_scaler: StandardScaler1D,
        train_cfg: TrainConfig,
        device: torch.device,
        curve_len: int,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.train_cfg = train_cfg
        self.device = device
        self.curve_len = curve_len

        ensure_dir(train_cfg.ckpt_dir)
        ensure_dir(train_cfg.log_dir)
        ensure_dir(train_cfg.plot_dir)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # 日志 CSV 初始化
        self.log_csv_path = os.path.join(train_cfg.log_dir, "loss.csv")
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_mse,val_mse,lr\n")

        # 状态
        self.best_val = float("inf")
        self.best_ckpt_path = os.path.join(train_cfg.ckpt_dir, "best_model.pth")
        self.state_path = os.path.join(train_cfg.ckpt_dir, "trainer_state.pth")
        self.start_epoch = 1

        # 断点续训
        if train_cfg.resume:
            self._try_resume()

    def _try_resume(self):
        if os.path.exists(self.state_path):
            print(f"[续训] 加载状态：{self.state_path}")
            state = torch.load(self.state_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.best_val = state.get("best_val", float("inf"))
            self.start_epoch = state.get("epoch", 1)
        else:
            # 尝试加载 best
            if os.path.exists(self.best_ckpt_path):
                print(f"[续训] 找到最佳模型：{self.best_ckpt_path}，将基于它继续训练（优化器重新开始）。")
                ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt)

    def _save_state(self, epoch: int):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val": self.best_val,
        }
        torch.save(state, self.state_path)

    def _save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(self.train_cfg.ckpt_dir, f"epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[保存] 检查点：{ckpt_path}")

    def _save_best(self):
        torch.save(self.model.state_dict(), self.best_ckpt_path)
        print(f"[保存] 最佳模型：{self.best_ckpt_path}")

    def train(self):
        patience_counter = 0
        for epoch in range(self.start_epoch, self.train_cfg.epochs + 1):
            t0 = time.time()
            train_mse = self._run_one_epoch(self.train_loader, train=True)
            val_mse = self._run_one_epoch(self.val_loader, train=False)

            # 调度器更新
            self.scheduler.step(val_mse)

            # 日志记录
            lr_now = self.optimizer.param_groups[0]["lr"]
            with open(self.log_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_mse:.6f},{val_mse:.6f},{lr_now:.8f}\n")

            dt = time.time() - t0
            print(f"[Epoch {epoch}/{self.train_cfg.epochs}] train_mse={train_mse:.6e} | val_mse={val_mse:.6e} | lr={lr_now:.2e} | {dt:.1f}s")

            # 最佳与早停
            improved = val_mse < self.best_val - 1e-8
            if improved:
                self.best_val = val_mse
                self._save_best()
                patience_counter = 0
            else:
                patience_counter += 1

            # 定期保存检查点
            if epoch % self.train_cfg.save_every == 0:
                self._save_checkpoint(epoch)

            # 保存训练状态（便于断点续训）
            self._save_state(epoch)

            if patience_counter >= self.train_cfg.patience:
                print(f"[早停] 验证集未提升 {self.train_cfg.patience} 次，停止训练。")
                break

        print("[训练] 完成。")

    def _run_one_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(self.device)  # [B, N]
            yb = yb.to(self.device)  # [B, 2]

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(xb)     # [B, 2]
            loss = self.criterion(pred, yb)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        return total_loss / max(1, total_count)

    @torch.no_grad()
    def test_and_visualize(self):
        """加载最佳模型进行测试评估与可视化。"""
        if os.path.exists(self.best_ckpt_path):
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))
            print(f"[测试] 已加载最佳模型：{self.best_ckpt_path}")
        else:
            print("[测试] 未找到最佳模型，使用当前权重。")

        self.model.eval()
        preds_norm = []
        trues_norm = []

        for xb, yb in self.test_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            pred = self.model(xb)
            preds_norm.append(pred.cpu().numpy())
            trues_norm.append(yb.cpu().numpy())

        preds_norm = np.concatenate(preds_norm, axis=0)  # (N,2) 归一化空间
        trues_norm = np.concatenate(trues_norm, axis=0)

        # 反变换回真实物理量
        preds_real = self.output_scaler.inverse_transform(preds_norm)  # (N,2)
        trues_real = self.output_scaler.inverse_transform(trues_norm)

        # 计算指标
        metrics = compute_metrics(preds_real, trues_real)
        print("[测试] 评估指标（真实物理量）：", metrics)

        # 保存指标到 JSON
        with open(os.path.join(self.train_cfg.log_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 可视化：预测 vs 真实（散点）
        self._plot_scatter(preds_real, trues_real, names=["sigma_y", "n"])

        # （可选）基于预测参数生成“近似曲线”与真实输入曲线做对比
        # 说明：真实输入曲线 = test_loader 中 xb（已标准化），需反标准化才能回到原始载荷尺度。
        #      由于我们只有曲线->参数的反问题，这里用近似前向模型生成曲线，仅供示意。
        try:
            self._plot_curves_comparison(preds_real, trues_real)
        except Exception as e:
            print(f"[曲线可视化] 跳过，原因：{e}")

    def _plot_scatter(self, preds: np.ndarray, trues: np.ndarray, names: List[str]):
        ensure_dir(self.train_cfg.plot_dir)
        for j, name in enumerate(names):
            plt.figure()
            plt.scatter(trues[:, j], preds[:, j], s=18, alpha=0.7)
            minv = float(min(trues[:, j].min(), preds[:, j].min()))
            maxv = float(max(trues[:, j].max(), preds[:, j].max()))
            plt.plot([minv, maxv], [minv, maxv], linestyle="--")
            plt.xlabel(f"真实 {name}")
            plt.ylabel(f"预测 {name}")
            plt.title(f"预测 vs 真实：{name}")
            out_path = os.path.join(self.train_cfg.plot_dir, f"scatter_{name}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[保存] 散点图：{out_path}")

    def _plot_curves_comparison(self, preds_real: np.ndarray, trues_real: np.ndarray, num_examples: int = 3):
        """
        随机抽取若干测试样本：
        - 原始输入曲线（反标准化后）作为“真实曲线”
        - 用预测参数通过 approx_forward_curve 生成“近似预测曲线”
        """
        ensure_dir(self.train_cfg.plot_dir)
        # 取一批测试样本的原始输入（需从 loader 再取一次以还原）
        xb_list = []
        for xb, _ in self.test_loader:
            xb_list.append(xb.numpy())
        xb_norm = np.concatenate(xb_list, axis=0)  # (N, seq_len)
        # 反标准化输入曲线：注意 input_scaler 是按列（位置）标准化的
        xb_real = self.input_scaler.inverse_transform(xb_norm)  # (N, seq_len)

        N = xb_real.shape[0]
        idxs = np.random.choice(N, size=min(num_examples, N), replace=False)
        for k, idx in enumerate(idxs):
            true_curve = xb_real[idx]  # (seq_len,)
            sigma_y_pred, n_pred = preds_real[idx, 0], preds_real[idx, 1]
            # 通过简化前向模型生成“近似预测曲线”
            pred_curve = approx_forward_curve(float(sigma_y_pred), float(n_pred), length=self.curve_len)

            plt.figure()
            plt.plot(true_curve, label="真实曲线（输入）")
            plt.plot(pred_curve, linestyle="--", label="近似预测曲线（示意）")
            plt.xlabel("等间距位移采样点索引")
            plt.ylabel("载荷值")
            plt.title(f"样本 {idx}：真实 vs 近似预测 曲线\n(注意：预测曲线仅为示意，可替换为真实前向模型)")
            plt.legend()
            out_path = os.path.join(self.train_cfg.plot_dir, f"curve_compare_{k}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[保存] 曲线对比图：{out_path}")


# ------------------------------
# 构建数据管线（含标准化拟合、划分、DataLoader）
# ------------------------------

def build_dataloaders(df: pd.DataFrame, data_cfg: DataConfig, train_cfg: TrainConfig) -> Tuple[
    DataLoader, DataLoader, DataLoader, StandardScaler1D, StandardScaler1D, int
]:
    # 先粗处理（不做标准化），以便计算 scaler：
    base_dataset = ExcelCurveDataset(df, data_cfg, input_scaler=None, output_scaler=None)
    x_all = base_dataset.x  # (N, 202)
    y_all = base_dataset.y  # (N, 2)

    # 标准化：在“训练+验证”集上拟合更稳妥；此处先以全体拟合，随后严格在 loader 中分割
    input_scaler = StandardScaler1D().fit(x_all)    # 列向标准化（每个采样点各自标准化）
    output_scaler = StandardScaler1D().fit(y_all)   # 两个输出各自标准化

    # 用 scaler 重新构建数据集（映射到标准化空间）
    full_dataset = ExcelCurveDataset(df, data_cfg, input_scaler=input_scaler, output_scaler=output_scaler)

    # 划分：80% 训练 + 20% 测试
    total_len = len(full_dataset)
    train_len = int(total_len * data_cfg.train_ratio)
    test_len = total_len - train_len
    train_dataset, test_dataset = random_split(
        full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(data_cfg.random_seed)
    )

    # 再从训练集切出 10% 做验证
    val_len = int(len(train_dataset) * data_cfg.val_within_train_ratio)
    new_train_len = len(train_dataset) - val_len
    train_dataset, val_dataset = random_split(
        train_dataset, [new_train_len, val_len], generator=torch.Generator().manual_seed(data_cfg.random_seed + 1)
    )

    print(f"[划分] 总样本={total_len} -> 训练={len(train_dataset)} 验证={len(val_dataset)} 测试={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, drop_last=False)

    seq_len = x_all.shape[1]
    return train_loader, val_loader, test_loader, input_scaler, output_scaler, seq_len


def save_scalers(input_scaler: StandardScaler1D, output_scaler: StandardScaler1D, path: str):
    ensure_dir(os.path.dirname(path))
    d = {"input_scaler": input_scaler.to_dict(), "output_scaler": output_scaler.to_dict()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"[保存] 标准化参数：{path}")


def load_scalers(path: str) -> Tuple[StandardScaler1D, StandardScaler1D]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    input_scaler = StandardScaler1D.from_dict(d["input_scaler"])
    output_scaler = StandardScaler1D.from_dict(d["output_scaler"])
    return input_scaler, output_scaler


# ------------------------------
# 主流程
# ------------------------------

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[设备] 使用设备：{device}")

    # 配置
    data_cfg = DataConfig(
        excel_path=args.excel_path,
        expected_cols=204,
        start_curve_col=3,
        end_curve_col=204,
        train_ratio=args.train_ratio,
        val_within_train_ratio=args.val_ratio_within_train,
        random_seed=args.seed,
        clip_outliers=not args.no_clip,
        clip_quantiles=(args.clip_low, args.clip_high),
        dropna=not args.keep_na
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        save_every=args.save_every,
        resume=args.resume,
        num_workers=args.num_workers
    )
    model_cfg = ModelConfig(
        width=args.width,
        modes=args.modes,
        layers=args.layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout
    )

    # 读取 Excel
    df = load_excel_dataframe(data_cfg.excel_path)

    # 构建 DataLoader 与标准化
    train_loader, val_loader, test_loader, input_scaler, output_scaler, seq_len = build_dataloaders(df, data_cfg, train_cfg)

    # 保存 scaler 供推理使用
    save_scalers(input_scaler, output_scaler, os.path.join(train_cfg.ckpt_dir, "scalers.json"))

    # 模型
    model = FNO1d(
        seq_len=seq_len,
        in_channels=1,
        width=model_cfg.width,
        modes=model_cfg.modes,
        layers=model_cfg.layers,
        mlp_hidden=model_cfg.mlp_hidden,
        out_dim=2,
        dropout=model_cfg.dropout
    )

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        train_cfg=train_cfg,
        device=device,
        curve_len=seq_len
    )

    # 训练
    trainer.train()

    # 中文字体设置
    setup_chinese_font()
    # 测试与可视化
    trainer.test_and_visualize()

    print("[完成] 全流程执行结束。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNO-1D 固体力学参数反演（从载荷-位移曲线反演 σY 与 n）")
    # 数据与划分
    parser.add_argument("--excel_path", type=str, default="Data.xlsx", help="Excel 文件路径（默认与脚本同目录）")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="训练集占比（默认 0.80）；剩余为测试集")
    parser.add_argument("--val_ratio_within_train", type=float, default=0.10, help="从训练集中切出的验证集比例（默认 0.10）")

    # 训练相关
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数（默认 1000）")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小（默认 16）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（默认 1e-3）")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减（默认 1e-6）")
    parser.add_argument("--patience", type=int, default=50, help="早停容忍（默认 50）")
    parser.add_argument("--save_every", type=int, default=100, help="每隔多少 epoch 保存检查点（默认 100）")
    parser.add_argument("--resume", action="store_true", help="断点续训")

    # 模型超参
    parser.add_argument("--width", type=int, default=64, help="FNO 通道宽度（默认 64）")
    parser.add_argument("--modes", type=int, default=16, help="频域保留的低频模数（默认 16）")
    parser.add_argument("--layers", type=int, default=4, help="谱卷积层数（默认 4）")
    parser.add_argument("--mlp_hidden", type=int, default=128, help="读出层隐藏维度（默认 128）")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout（默认 0.0）")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存目录")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="可视化图保存目录")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的 num_workers（Windows 建议 0）")

    # 清洗参数
    parser.add_argument("--no_clip", action="store_true", help="不开启分位数裁剪")
    parser.add_argument("--clip_low", type=float, default=0.005, help="低分位（默认 0.005）")
    parser.add_argument("--clip_high", type=float, default=0.995, help="高分位（默认 0.995）")
    parser.add_argument("--keep_na", action="store_true", help="保留缺失值（默认删除含 NaN 的行）")

    args = parser.parse_args()
    main(args)
