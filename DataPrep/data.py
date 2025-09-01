
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StandardScaler1D:
    """
    一个极简版的“标准化器”（类似 sklearn.StandardScaler），
    作用：对每个样本的特征向量做标准化：z = (x - mean) / std
    - mean_ / std_ 的形状都是 [1, D]，D 为特征维度（列数）
    - 注意：在调用 transform / inverse_transform 前，必须先 fit()
    """
    def __init__(self):
        self.mean_ = None   # 保存按列计算的均值，形状 [1, D]
        self.std_  = None   # 保存按列计算的标准差，形状 [1, D]

    def fit(self, x: np.ndarray):
        """
        根据数据 x 估计均值和标准差（按列）。
        - 将 x 转为 float32，并 reshape 为 [N, D]（N=样本数，D=特征数）
        - std 加上 1e-8，避免除以 0 的数值问题
        返回自身，以便链式调用（如 scaler.fit(x).transform(x)）
        """
        x = np.asarray(x, dtype=np.float32)
        x = x.reshape(len(x), -1)  # 保证是 [N, D] 的二维数组
        self.mean_ = x.mean(0, keepdims=True)                   # [1, D]
        self.std_  = x.std(0, keepdims=True) + 1e-8             # [1, D]（防止为 0）
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        用已学习到的 mean_ / std_ 对新数据做标准化。
        要求：已先调用 fit()，否则 mean_/std_ 为空会报错。
        """
        x = np.asarray(x, dtype=np.float32).reshape(len(x), -1) # [N, D]
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        将标准化后的数据还原回原始尺度：x = z * std + mean
        （常用于：训练/推理后把模型输出从标准化空间映射回物理量）
        """
        x = np.asarray(x, dtype=np.float32).reshape(len(x), -1) # [N, D]
        return x * self.std_ + self.mean_

    def to_dict(self):
        """
        以 Python 字典形式导出均值与方差信息（可序列化成 JSON 保存到硬盘）
        .tolist() 是为了让 numpy 数组能被 JSON 正常写出
        """
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @staticmethod
    def from_dict(d):
        """
        从字典（通常是从 JSON 读回的内容）恢复一个 StandardScaler1D。
        """
        s = StandardScaler1D()
        s.mean_ = np.asarray(d["mean"], np.float32)  # [1, D]
        s.std_  = np.asarray(d["std"],  np.float32)  # [1, D]
        return s


def save_scalers(input_scaler: StandardScaler1D, output_scaler: StandardScaler1D, path: str | Path):
    """
    将“输入标准化器”和“输出标准化器”一起保存为一个 JSON 文件。
    常见用法：训练结束后把 in/out 的 scaler 一并持久化，推理时加载保持一致的尺度。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_scaler":  input_scaler.to_dict(),
                "output_scaler": output_scaler.to_dict()
            },
            f, indent=2, ensure_ascii=False
        )

def load_scalers(path: str | Path) -> Tuple[StandardScaler1D, StandardScaler1D]:
    """
    从 JSON 文件加载“输入/输出标准化器”各自的 mean/std，恢复为可用对象。
    返回： (input_scaler, output_scaler)
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return (
        StandardScaler1D.from_dict(d["input_scaler"]),
        StandardScaler1D.from_dict(d["output_scaler"])
    )


@dataclass
class DataConfig:
    """
    数据读取与拆分的配置项（dataclass 仅用于存放参数，便于管理与类型检查）

    字段说明
    --------
    excel_path : Excel 文件路径（训练/推理用到的原始数据表）
    expected_cols : 期望的总列数（用于基本校验，防止表头/列数不对）
    start_curve_col : 曲线数据在 Excel 中的起始列索引（通常按 1 开始的“人类习惯”，含参数列后从第3列开始）
    end_curve_col   : 曲线数据在 Excel 中的结束列索引（与起始列一样，为“包含端点”的 1 基索引）
    train_ratio : 训练集比例（0~1 之间），其余作为测试集
    val_within_train_ratio : 从训练集中再切出一部分做验证集的比例（0~1 之间）
    random_seed : 随机种子（保证可复现的划分）
    dropna : 是否在读入后丢弃包含缺失值的样本行
    clip_outliers : 是否对曲线列做分位点裁剪（鲁棒去极端值）
    clip_low / clip_high : 分位裁剪的下/上界（例如 0.5% 与 99.5%）
    """
    excel_path: str = "Data.xlsx"
    expected_cols: int = 204
    start_curve_col: int = 3
    end_curve_col: int = 204
    train_ratio: float = 0.80
    val_within_train_ratio: float = 0.10
    random_seed: int = 42
    dropna: bool = True
    clip_outliers: bool = True
    clip_low: float = 0.005
    clip_high: float = 0.995


def load_excel_dataframe(path: str | Path) -> pd.DataFrame:
    # 统一把输入转为 Path 对象，先检查文件是否存在
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 Excel：{path}")

    # 尝试读取；失败时给出更有指导性的报错信息
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(
            f"读取 Excel 失败：{e}（请确认安装 openpyxl 且文件未被占用）"
        )

class ExcelCurveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DataConfig,
                 input_scaler: Optional[StandardScaler1D] = None,
                 output_scaler: Optional[StandardScaler1D] = None):
        if df.shape[1] < cfg.expected_cols:
            raise ValueError(f"列数不足：期望≥{cfg.expected_cols}，实际 {df.shape[1]}")
        df = df.iloc[:, :cfg.expected_cols].copy()
        cols = ["sigma_y","n"] + [f"curve_{i}" for i in range(1, cfg.end_curve_col-cfg.start_curve_col+2)]
        df.columns = cols
        if cfg.dropna: df = df.dropna()
        if cfg.clip_outliers:
            curve_cols = [c for c in df.columns if c.startswith("curve_")]
            ql = df[curve_cols].quantile(cfg.clip_low); qh = df[curve_cols].quantile(cfg.clip_high)
            df[curve_cols] = df[curve_cols].clip(lower=ql, upper=qh, axis=1)
        self.df = df.reset_index(drop=True)
        x = self.df[[c for c in self.df.columns if c.startswith("curve_")]].values.astype(np.float32)
        y = self.df[["sigma_y","n"]].values.astype(np.float32)
        if input_scaler is not None: x = input_scaler.transform(x)
        if output_scaler is not None: y = output_scaler.transform(y)
        self.x, self.y = x, y
    def __len__(self): return len(self.df)
    def __getitem__(self, i): 
        import torch
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

def build_dataloaders(df: pd.DataFrame, dcfg: DataConfig, batch_size: int, num_workers: int):
    """
    将 DataFrame -> 标准化 -> 数据集拆分 -> DataLoader 构建。
    返回（train_loader, val_loader, test_loader, in_scaler, out_scaler, seq_len）
    """

    # 1) 先用“未标准化”的临时 Dataset 拿到所有 X/Y，用于拟合 scaler（按列求 mean/std）
    base = ExcelCurveDataset(df, dcfg, None, None)
    x_all, y_all = base.x, base.y

    # 2) 分别对输入曲线与输出参数拟合标准化器（训练时的统计量）
    in_scaler  = StandardScaler1D().fit(x_all)
    out_scaler = StandardScaler1D().fit(y_all)

    # 3) 用拟合好的 scaler 重建“正式”的 Dataset（内部将自动 transform 到标准化空间）
    full = ExcelCurveDataset(df, dcfg, in_scaler, out_scaler)

    # 4) 按配置比例切分 train/test；使用固定随机种子保证可复现
    total = len(full)
    tr_len = int(total * dcfg.train_ratio)   # 训练集样本数
    te_len = total - tr_len                  # 测试集样本数（剩余）
    gen = torch.Generator().manual_seed(dcfg.random_seed)
    from torch.utils.data import random_split
    train_set, test_set = random_split(full, [tr_len, te_len], generator=gen)

    # 5) 再从训练集内部切出验证集（val），同样用固定种子保证复现
    val_len = int(len(train_set) * dcfg.val_within_train_ratio)
    new_tr_len = len(train_set) - val_len
    train_set, val_set = random_split(
        train_set, [new_tr_len, val_len],
        generator=torch.Generator().manual_seed(dcfg.random_seed + 1)
    )

    # 6) 构建 DataLoader
    # - 训练集 shuffle=True 打乱样本；验证/测试不打乱
    # - drop_last=False 保留不足一个 batch 的尾样本
    # - Windows/小数据上建议 num_workers=0（已由外部传入）
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # 7) 记录序列长度（曲线的采样点数），供模型构造/校验
    seq_len = x_all.shape[1]

    # 返回：三类 DataLoader + 两个 scaler + 序列长度
    return train_loader, val_loader, test_loader, in_scaler, out_scaler, seq_len

