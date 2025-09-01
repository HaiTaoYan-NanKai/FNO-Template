
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

def setup_chinese_font():
    candidates = ["Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei","SimHei","PingFang SC",
                  "WenQuanYi Zen Hei","Songti SC","STHeiti"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for n in candidates:
        if n in available:
            rcParams["font.sans-serif"] = [n]; break
    rcParams["axes.unicode_minus"] = False

def approx_forward_curve(sigma_y: float, n: float, length: int = 202) -> np.ndarray:
    t = np.linspace(0, 1, length, dtype=np.float32)
    half = length // 2; load = np.zeros_like(t)
    E = max(1.0, sigma_y / 0.01)
    for i in range(half):
        strain = t[i]; stress = E * strain
        if stress < sigma_y:
            load[i] = stress
        else:
            k = sigma_y
            load[i] = sigma_y + k * max(0.0, strain - sigma_y / E) ** max(n, 0.01)
    end_val = load[half - 1]
    for i in range(half, length):
        a = (i - half) / max(1, (length - half - 1))
        load[i] = (1 - a) * end_val
    return load

def plot_scatter(save_dir: str, preds: np.ndarray, trues: np.ndarray, names: List[str]):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(names):
        plt.figure()
        plt.scatter(trues[:, j], preds[:, j], s=18, alpha=0.7)
        mn = float(min(trues[:, j].min(), preds[:, j].min()))
        mx = float(max(trues[:, j].max(), preds[:, j].max()))
        plt.plot([mn, mx], [mn, mx], ls="--")
        plt.xlabel(f"真实 {name}"); plt.ylabel(f"预测 {name}")
        plt.title(f"预测 vs 真实：{name}")
        p = Path(save_dir)/f"scatter_{name}.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()

def plot_curves_compare(save_dir: str, true_curves: np.ndarray,
                        preds_params: np.ndarray, curve_len: int, num_examples: int = 3):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    N = true_curves.shape[0]
    import numpy as np
    idxs = np.random.choice(N, size=min(num_examples, N), replace=False)
    for k, idx in enumerate(idxs):
        tcurve = true_curves[idx]
        s, n = float(preds_params[idx, 0]), float(preds_params[idx, 1])
        pcurve = approx_forward_curve(s, n, curve_len)
        plt.figure()
        plt.plot(tcurve, label="真实曲线（输入）")
        plt.plot(pcurve, ls="--", label="近似预测曲线（示意）")
        plt.xlabel("等间距位移采样点索引"); plt.ylabel("载荷值")
        plt.title(f"样本 {idx}：真实 vs 近似预测 曲线\n(注意：预测曲线仅为示意，可替换为真实前向模型)")
        plt.legend(); p = Path(save_dir)/f"curve_compare_{k}.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
