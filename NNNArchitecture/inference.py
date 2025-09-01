
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from Allocation.config import ROOT, Config, get_device
from DataPrep.data import (DataConfig, load_excel_dataframe, ExcelCurveDataset,
                           load_scalers)
from NNNArchitecture.model import FNO1d
from AnalyzeVisualize.visualisation import setup_chinese_font, plot_scatter

def main():
    # === 命令行参数 ===
    # 例：python -m NNNArchitecture.inference --excel_path ./Data.xlsx --config ./Allocation/config.json
    p = argparse.ArgumentParser("FNO-1D 推理")
    p.add_argument("--config", type=str, default=str(ROOT/"Allocation"/"config.json"))  # 配置文件路径
    p.add_argument("--excel_path", type=str, default=None, help="（可覆盖）Excel 路径")      # 可覆盖默认 Excel
    args = p.parse_args()

    # === 读取配置与设备选择 ===
    cfg = Config.load(args.config)         # 从 JSON 读取所有超参数/路径
    device = get_device()                  # 自动选择 'cuda' 或 'cpu'
    excel_path = args.excel_path or cfg.excel_path  # 若命令行未指定，则用配置里的路径

    # === 构造数据读取配置并读取 Excel ===
    # 只做推理所需的预处理配置（列数、裁剪、缺失处理等）
    dcfg = DataConfig(excel_path=excel_path,
                      expected_cols=cfg["data"]["expected_cols"],
                      start_curve_col=cfg["data"]["start_curve_col"],
                      end_curve_col=cfg["data"]["end_curve_col"],
                      dropna=cfg["data"]["dropna"],
                      clip_outliers=cfg["data"]["clip_outliers"],
                      clip_low=cfg["data"]["clip_low"],
                      clip_high=cfg["data"]["clip_high"])
    df = load_excel_dataframe(dcfg.excel_path)  # 读取 Excel -> DataFrame

    # === 载入训练期保存的 scaler 与模型结构 ===
    ckpt_dir = cfg["train"]["ckpt_dir"]
    in_scaler, out_scaler = load_scalers(Path(ckpt_dir)/"scalers.json")  # 输入/输出标准化器（与训练一致）

    # 模型实例化：序列长度由数据列范围决定；其它超参来自配置
    seq_len = dcfg.end_curve_col - dcfg.start_curve_col + 1
    model = FNO1d(seq_len=seq_len,
                  width=cfg["model"]["width"],
                  modes=cfg["model"]["modes"],
                  layers=cfg["model"]["layers"],
                  mlp_hidden=cfg["model"]["mlp_hidden"],
                  dropout=cfg["model"]["dropout"]).to(device)

    # 加载“最佳模型”参数，并切到评估模式
    best_path = Path(ckpt_dir)/"best_model.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # === 构建仅用于推理的 Dataset ===
    # 这里把训练期的 in/out scaler 传入，保证与训练时同样的标准化方式
    ds = ExcelCurveDataset(df, dcfg, input_scaler=in_scaler, output_scaler=out_scaler)

    # === 前向推理（不需要梯度） ===
    x = torch.from_numpy(ds.x).to(device)  # 标准化后的输入曲线，形状 [N, seq_len]
    with torch.no_grad():
        pred_n = model(x).cpu().numpy()    # 模型输出（仍在“标准化空间”）

    # 反标准化回到真实物理量尺度；若 df 中含真值（σy、n），也一并反变换便于对比
    preds = out_scaler.inverse_transform(pred_n)  # 预测（真实尺度）
    trues = out_scaler.inverse_transform(ds.y)    # 真值（真实尺度），如果部署场景没有真值，这行可忽略

    # === 保存推理结果为 CSV ===
    out_csv = Path(cfg["train"]["log_dir"])/"inference.csv"
    header = "pred_sigma_y,pred_n,true_sigma_y,true_n"
    # 将预测与真值按行写出；若没有真值，可只写预测列
    lines = [header] + [f"{a[0]},{a[1]},{b[0]},{b[1]}" for a, b in zip(preds, trues)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"[输出] 预测结果：{out_csv}")

    # === 可视化（可选） ===
    # 若有真值，可绘制 预测 vs 真实 的散点图；异常不影响主流程，故用 try 包裹
    try:
        setup_chinese_font()  # 解决中文显示问题（可按需安装中文字体）
        plot_scatter(cfg["train"]["plot_dir"], preds, trues, names=["sigma_y","n"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
