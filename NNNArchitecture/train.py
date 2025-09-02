# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from Allocation.config import ROOT, Config, ensure_dir, set_seed, get_device
from Allocation.logger import get_logger
from DataPrep.data import (DataConfig, load_excel_dataframe, build_dataloaders,
                           save_scalers, StandardScaler1D)
from NNNArchitecture.model import FNO1d
from NNNArchitecture.loss import compute_metrics
from AnalyzeVisualize.visualisation import setup_chinese_font, plot_scatter, plot_curves_compare

def train_one_epoch(model, loader, device, optimizer, criterion):
    """
    训练单个 epoch，并返回该 epoch 的“平均损失”(per-sample average loss)。

    参数
    ----
    model      : nn.Module            # 待训练的模型
    loader     : DataLoader           # 训练集的数据迭代器，逐批返回 (xb, yb)
    device     : torch.device         # 训练设备（'cuda' 或 'cpu'）
    optimizer  : torch.optim.Optimizer# 优化器（如 Adam）
    criterion  : callable             # 损失函数（如 nn.MSELoss）

    返回
    ----
    float：本 epoch 在训练集上的平均损失（按样本数加权的平均值）
    """
    model.train()              # 进入“训练模式”（启用 Dropout/BatchNorm 的训练行为）
    total, cnt = 0.0, 0        # total 累加“加权批损失”，cnt 累加样本总数

    for xb, yb in loader:      # 遍历每个小批次
        # 将数据搬到目标设备（GPU/CPU）
        xb, yb = xb.to(device), yb.to(device)

        # 1) 清梯度：将上一次迭代的梯度清空（置为 None 更省显存/更快）
        optimizer.zero_grad(set_to_none=True)

        # 2) 前向传播：得到预测值
        pred = model(xb)

        # 3) 计算损失：pred 与 yb 的误差（如 MSE）
        loss = criterion(pred, yb)

        # 4) 反向传播：基于当前 loss 计算所有可训练参数的梯度
        loss.backward()

        # 5) （可选）梯度裁剪：按全局 L2 范数将梯度上限设为 5，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 6) 参数更新：优化器根据梯度对模型参数做一步更新
        optimizer.step()

        # 记录本批的样本数与损失（按样本数加权，便于得到总体平均）
        b = xb.size(0)                    # 当前批的样本数
        total += loss.item() * b          # 累加加权损失
        cnt   += b                        # 累加样本数

    # 返回“按样本数平均”的损失；max(1, cnt) 防止空数据集时除零
    return total / max(1, cnt)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    """
    评估单个 epoch（验证/测试阶段），返回“平均损失”(per-sample average loss)。

    说明
    ----
    - @torch.no_grad(): 关闭梯度计算与保存，减少显存占用、加快推理（评估时不需要反传）。
    - model.eval():     切换到评估模式（如关闭 Dropout、固定 BatchNorm 的均值/方差）。
    - 不做 optimizer.step() 与 zero_grad()：评估阶段只前向与计算指标。
    """
    model.eval()             # 评估模式（影响 Dropout/BatchNorm 等）
    total, cnt = 0.0, 0      # total 累加“加权批损失”，cnt 累加样本总数

    for xb, yb in loader:    # 逐批评估
        # 搬到目标设备（GPU/CPU）
        xb, yb = xb.to(device), yb.to(device)

        # 前向传播得到预测；评估阶段只算前向与损失
        pred = model(xb)
        loss = criterion(pred, yb)   # 如 nn.MSELoss 等

        # 按样本数加权汇总，便于得到全数据集的平均损失
        b = xb.size(0)
        total += loss.item() * b
        cnt   += b

    # 返回按样本数平均的损失；max(1, cnt) 防止极端情况下除零
    return total / max(1, cnt)


def main():
    parser = argparse.ArgumentParser("FNO-1D 训练")
    parser.add_argument("--config", type=str, default=str(ROOT/"Allocation"/"config.json"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed); device = get_device()
    log_dir = cfg["train"]["log_dir"]; ensure_dir(log_dir)
    logger = get_logger(log_dir, "train")
    logger.info(f"设备: {device}")

    # ----- Data：数据准备阶段 -----
    # 构建 DataConfig 配置对象，把 config.json 中关于数据的超参数集中传入
    # - excel_path: Excel 数据文件路径
    # - expected_cols: 期望的总列数（用于简单校验，防止 Excel 列数不对）
    # - start_curve_col / end_curve_col: 曲线数据在 Excel 中的起始/结束列（包含端点）
    # - train_ratio: 训练集比例（剩下的作为测试集）
    # - val_within_train_ratio: 验证集在训练集中的比例
    # - random_seed: 随机种子，保证数据划分可复现
    # - dropna: 是否丢弃包含缺失值的行
    # - clip_outliers: 是否对曲线做分位裁剪（去除极端值）
    # - clip_low / clip_high: 裁剪的上下分位点（如 0.5% 与 99.5%）
    dcfg = DataConfig(
        excel_path=cfg.excel_path,
        expected_cols=cfg["data"]["expected_cols"],
        start_curve_col=cfg["data"]["start_curve_col"],
        end_curve_col=cfg["data"]["end_curve_col"],
        train_ratio=cfg["data"]["train_ratio"],
        val_within_train_ratio=cfg["data"]["val_ratio_within_train"],
        random_seed=cfg.seed,
        dropna=cfg["data"]["dropna"],
        clip_outliers=cfg["data"]["clip_outliers"],
        clip_low=cfg["data"]["clip_low"],
        clip_high=cfg["data"]["clip_high"]
    )

    # 读取 Excel 文件为 pandas.DataFrame
    # - 内部会调用 openpyxl 引擎
    # - 若文件不存在或被占用会抛出异常
    df = load_excel_dataframe(dcfg.excel_path)

    # 根据配置与 DataFrame 构建 DataLoader
    # - build_dataloaders 会做几件事：
    #   1. 先用数据 fit 输入/输出的 StandardScaler（保存 mean/std）
    #   2. 重建 Dataset，把 X/Y 转换到标准化空间
    #   3. 按比例切分 train/val/test
    #   4. 返回三个 DataLoader，保证训练集 shuffle，验证/测试不打乱
    #   5. 返回 in_scaler / out_scaler，方便后续保存与推理时反标准化
    #   6. 返回序列长度 seq_len，供模型构造时使用
    tl, vl, te, in_scaler, out_scaler, seq_len = build_dataloaders(
        df,
        dcfg,
        cfg["train"]["batch_size"],   # 每个 batch 样本数
        cfg["train"]["num_workers"]   # DataLoader 的工作进程数（0 表示主进程）
    )


    ckpt_dir = cfg["train"]["ckpt_dir"]; ensure_dir(ckpt_dir)
    save_scalers(in_scaler, out_scaler, Path(ckpt_dir)/"scalers.json")

    # ----- Model -----
    model = FNO1d(seq_len=seq_len,
                  width=cfg["model"]["width"],
                  modes=cfg["model"]["modes"],
                  layers=cfg["model"]["layers"],
                  mlp_hidden=cfg["model"]["mlp_hidden"],
                  dropout=cfg["model"]["dropout"]).to(device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # 学习率调度器：当验证指标长期没有改进时，自动把学习率降低（帮助跳出“平台期”）
    sch = optim.lr_scheduler.ReduceLROnPlateau(
        opt,          # 需要被调度的优化器（如 Adam）
        mode="min",   # 监控的指标越小越好（通常用验证集损失 val_loss）
        factor=0.5,   # 触发时将当前学习率乘以 0.5（减半）
        patience=10,  # 连续 10 个 epoch 无改进才会降低学习率
        verbose=True  # 触发时在控制台打印提示信息
    )

    # ----- Logging CSV -----
    csv_path = Path(log_dir)/"loss.csv"
    if not csv_path.exists(): csv_path.write_text("epoch,train_mse,val_mse,lr\n", encoding="utf-8")

    # ----- Resume：断点续训/状态恢复 -----
    # 初始化“历史最优验证损失”和“起始 epoch”
    best_val, start_epoch = float("inf"), 1

    # 仅存“最佳模型权重”的路径（用于推理/评估）
    best_path = Path(ckpt_dir) / "best_model.pth"

    # 存放“训练器完整状态”的路径（用于继续训练：含 model/optimizer/scheduler/epoch 等）
    state_path = Path(ckpt_dir) / "trainer_state.pth"

    # 若配置中开启了断点续训且状态文件存在，则从该状态恢复
    if cfg["train"]["resume"] and state_path.exists():
        # 读取上次保存的训练状态；map_location 确保无论上次在 CPU/GPU 保存，
        # 这次都能映射到当前 device（避免设备不匹配报错）
        s = torch.load(state_path, map_location=device)

        # 恢复模型参数、优化器和学习率调度器的内部状态
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sch.load_state_dict(s["sch"])

        # 恢复历史最优验证损失与已训练到的 epoch（键缺失时使用默认值）
        best_val = s.get("best_val", best_val)
        start_epoch = s.get("epoch", 1)

        # 打印提示，便于在日志中确认从第几个 epoch 续上
        logger.info(f"从断点续训：epoch={start_epoch}")



   # ----- Train Loop：主训练循环 -----
    patience, pc = cfg["train"]["patience"], 0   # patience=允许“验证不提升”的最大轮数；pc=当前已连续未提升的轮数
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        t0 = time.time()

        # 1) 训练一个 epoch（返回训练集平均损失）
        tr = train_one_epoch(model, tl, device, opt, crit)

        # 2) 在验证集上评估（返回验证集平均损失）
        va = eval_epoch(model, vl, device, crit)

        # 3) 把“验证损失”喂给学习率调度器；若长期无提升，则自动降低学习率
        sch.step(va)

        # 4) 记录当前学习率（从优化器里读出来）
        lr_now = opt.param_groups[0]["lr"]

        # 5) 将本轮的 train/val loss 与 lr 追加写入 CSV（便于画曲线）
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr:.6f},{va:.6f},{lr_now:.8f}\n")

        # 6) 控制台与日志文件打印本轮信息（科学计数法 & 本轮耗时）
        logger.info(
            f"[{epoch}/{cfg['train']['epochs']}] "
            f"train={tr:.6e} | val={va:.6e} | lr={lr_now:.2e} | {time.time()-t0:.1f}s"
        )

        # 7) 判断验证集是否“显著提升”（阈值 1e-8），若提升则保存“最佳模型”并清零计数
        improved = va < best_val - 1e-8
        if improved:
            best_val = va
            torch.save(model.state_dict(), best_path)  # 仅保存权重（推理/最终评估用）
            pc = 0
            logger.info(f"保存最佳模型 -> {best_path}")
        else:
            pc += 1  # 未提升则累计“无提升”轮数

        # 8) 周期性保存检查点（按配置的 save_every），便于中途回溯/恢复
        if epoch % cfg["train"]["save_every"] == 0:
            ep_path = Path(ckpt_dir) / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), ep_path)
            logger.info(f"保存检查点 -> {ep_path}")

        # 9) 保存“训练器状态”（断点续训用）：包含 epoch、模型/优化器/调度器状态与 best_val
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "sch":   sch.state_dict(),
                "best_val": best_val
            },
            state_path
        )

        # 10) 早停：若连续 pc 次验证都未提升（pc >= patience），提前停止训练
        if pc >= patience:
            logger.info(f"早停：验证集 {patience} 次未提升")
            break


    # ----- Test & Visualize：测试与可视化 -----
    # 若存在“最佳模型”权重，则先加载它再做评估（以最佳泛化性能作报告）
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()                          # 切换到评估模式（关闭 Dropout/固定 BN 统计）
    preds_n, trues_n, xb_n = [], [], []   # 收集器：预测(标准化空间)、真值(标准化空间)、输入(标准化空间)

    with torch.no_grad():                 # 评估阶段不需要梯度，节省显存并加速
        for xb, yb in te:                 # te：测试集 DataLoader
            xb = xb.to(device)            # 只需前向，不必把 yb 放到 GPU（但下行取 numpy 也无所谓）
            pr = model(xb).cpu().numpy()  # 前向得到预测（标准化后的参数），搬回 CPU 转 numpy
            preds_n.append(pr)            # 追加本批预测（normalized）
            trues_n.append(yb.numpy())    # 追加本批真值（normalized）
            xb_n.append(xb.cpu().numpy()) # 追加本批输入曲线（normalized）

    # 将各批次沿 batch 维拼接成完整数组
    preds_n = np.concatenate(preds_n)     # 形如 [N_test, 2]，仍在“标准化空间”
    trues_n = np.concatenate(trues_n)     # 形如 [N_test, 2]
    xb_n    = np.concatenate(xb_n)        # 形如 [N_test, seq_len]

    # 反标准化回到“物理量/原始尺度”
    preds = out_scaler.inverse_transform(preds_n)  # 预测参数反变换（使用输出 scaler）
    trues = out_scaler.inverse_transform(trues_n)  # 真值参数反变换

    # 计算指标（MSE/MAE/R2 等），并保存到日志目录
    metrics = compute_metrics(preds, trues)
    Path(log_dir, "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 设置中文字体，避免绘图中文乱码
    setup_chinese_font()

    # 画散点：预测 vs 真实（σ_y 与 n 各一张图），保存到 plots/ 目录
    plot_scatter(cfg["train"]["plot_dir"], preds, trues, names=["sigma_y", "n"])

    # ---- 曲线对比（示意）----
    # 注意：xb_n 是“标准化后的输入曲线”。若要还原到原始曲线，应使用“输入的 scaler”做反变换。
    # 例如：true_curves = in_scaler.inverse_transform(xb_n)   （推荐做法）
    # 下面这行仅作“示意”占位，若 StandardScaler1D 未拟合会报错，请替换为上面的 in_scaler。
    true_curves = StandardScaler1D().inverse_transform(xb_n)  # 注意：这里只用于画图示意

    # 用近似前向模型把预测的 (σ_y, n) 生成“预测曲线”，与真实输入曲线做可视化对比（抽样 num_examples 条）
    '''
    plot_curves_compare(
        cfg["train"]["plot_dir"],
        true_curves,
        preds,
        curve_len=xb_n.shape[1],
        num_examples=3
    )
    '''
    logger.info("[完成] 训练与测试结束。")


if __name__ == "__main__":
    main()
