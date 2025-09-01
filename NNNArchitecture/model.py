
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        参数说明：
            in_channels : 输入通道数 I（进入本层的特征图条数）
            out_channels: 输出通道数 O（本层要产生的特征图条数）
            modes       : 保留的傅里叶模态/频率桶个数 K（只学习前 K 个低频系数）
        作用概览：
            在频域里为每个频率 k 学一张 I×O 的“复数权重矩阵” W(k)=br+ i·bi，
            用它把 I 个输入通道线性组合成 O 个输出通道。
        """
        super().__init__()  # 先初始化 nn.Module 的基础设施（参数注册、设备迁移等）
        # 记录超参数，forward 时会用到
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        # 初始化缩放因子：把随机权重的幅度变小，避免通道求和后输出过大导致训练不稳
        # 这里用 1/(I*O) 做一个简单而保守的缩放（也可选用 Xavier/He 初始化思路）
        scale = 1.0 / (in_channels * out_channels)
        # 频域权重的“实部”和“虚部”，形状均为 [I, O, K]：
        # - I: 输入通道，O: 输出通道，K: 保留的频率数（modes）
        # - nn.Parameter 表示“可训练参数”，会参与反向传播并被优化器更新
        # - 用标准正态 randn 初始化后再乘以 scale 将其缩小
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, a: torch.Tensor, br: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
        """
        在频域做一次复数线性变换（FNO 的谱卷积核心步骤）。

        形状约定：
            a  : [B, I, K]  —— rFFT 后的输入谱特征（复数）；B=批大小，I=输入通道数，K=频率模态数
            br : [I, O, K]  —— 可学习“复权重”的实部（浮点数）
            bi : [I, O, K]  —— 可学习“复权重”的虚部（浮点数）
            返回: [B, O, K]  —— 输出谱特征（复数）
        
        数学含义：
            对每个频率 k，都有一张 I×O 的复矩阵 W(k) = br(:,:,k) + i·bi(:,:,k)，
            本函数计算  a(:, :, k) × W(k)  的结果。
            复数乘法拆分公式：(ar + i·ai)(br + i·bi) = (ar·br - ai·bi) + i(ar·bi + ai·br)
        
        einsum 说明（"bik,iok->bok"）：
            在维度符号中 b=batch, i=in_channel, o=out_channel, k=mode。
            出现在输入但没出现在输出的维度（这里是 i）会被“乘后求和”，
            保留 b、o、k 维，得到形状 [B, O, K]。
        """

        # 实部 = ar·br  -  ai·bi
        # ar·br：对输入通道 i 做加权求和，得到每个 (b, o, k) 的实部贡献
        real = torch.einsum("bik,iok->bok", a.real, br) - torch.einsum("bik,iok->bok", a.imag, bi)
        # 虚部 = ar·bi  +  ai·br
        # 仍然是对输入通道 i 做加权求和，得到每个 (b, o, k) 的虚部贡献    
        imag = torch.einsum("bik,iok->bok", a.real, bi) + torch.einsum("bik,iok->bok", a.imag, br)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        # x: [B, C_in, N] 的时域实信号
        # 1) 正向 FFT：把时域信号变到频域，只保留非负频率（更省内存/计算）
        x_ft = torch.fft.rfft(x, n=N)
        # 2) 只在前 K 个低频上做线性变换，其余高频保持为 0（相当于低通/带宽控制）
        K = min(self.modes, x_ft.shape[-1])  # self.modes 是你设定要“保留/学习”的模态数
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.complex64, device=x.device)
        out_ft[:, :, :K] = self.compl_mul1d(x_ft[:, :, :K], self.weight_real[:, :, :K], self.weight_imag[:, :, :K])
        return torch.fft.irfft(out_ft, n=N)

class FNO1d(nn.Module):
    def __init__(self, 
             seq_len: int,                 # 输入序列长度 N（用于 forward 时做长度一致性检查）
             in_channels: int = 1,         # 原始输入通道数（如 1 条曲线 -> 1 通道）
             width: int = 64,              # 模型“宽度”：中间隐表示的通道数（提升表达力的关键超参）
             modes: int = 16,              # 参与学习的傅里叶低频模态个数（频域带宽/容量）
             layers: int = 4,              # 谱卷积块的堆叠层数（越多越深）
             mlp_hidden: int = 128,        # 输出头（MLP）中的隐藏层维度
             out_dim: int = 2,             # 最终输出维度（本任务中为 2：σ_y 与 n）
             dropout: float = 0.0):        # Dropout 概率（0 表示关闭）
        super().__init__()

        # 记录输入长度，forward 时会检查输入的实际长度是否等于 seq_len
        self.seq_len = seq_len

        # 1×1 一维卷积：仅在“通道维”上做线性映射，不改变长度维
        # 作用：把原始通道 in_channels “抬升/投影”到较宽的隐空间 width
        # 形状：[B, in_channels, N] → [B, width, N]
        self.input_proj = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)

        # 谱卷积层组：每层在频域内对前 modes 个低频进行线性变换（复权重），
        # 输入/输出通道都采用 width（即 I=O=width），共堆叠 layers 层
        # 使用 ModuleList 以确保子层被正确注册为可训练模块
        self.spectral_layers = nn.ModuleList([
        SpectralConv1d(width, width, modes) for _ in range(layers)
        ])

        # 逐点卷积层组（1×1 Conv1d）：与谱层并行/残差相加，提供时域的局部线性混合，
        # 常配合激活一起使用，提升非线性表达与稳定性
        self.pointwise_layers = nn.ModuleList([
        nn.Conv1d(width, width, kernel_size=1) for _ in range(layers)
        ])

        # 非线性激活：GELU（较 ReLU 平滑，常用于 Transformer/FNO 等）
        self.act = nn.GELU()

        # Dropout：若 p=0 则用 Identity 占位，减少条件分支开销
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 读出头（MLP）：将时/频域处理后的全局特征（后续会做 GAP 得到 [B, width]）
        # 映射到目标参数空间 out_dim
        # 结构：Linear(width→mlp_hidden) → GELU → Linear(mlp_hidden→out_dim)
        self.readout = nn.Sequential(
            nn.Linear(width, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, out_dim)
        )

        # 参数初始化：对 Conv/Linear 等权重做合适的初始化（如 Kaiming），
        # 有助于稳定训练与加快收敛（具体策略见 _init 的实现）
        self._init()

    def _init(self):
        """
        统一初始化本模型中不同类型子层的参数，保证训练稳定、收敛更快。
        做法：遍历所有子模块，针对 Conv1d 和 Linear 分别采用合适的 Kaiming(He) 初始化；
            其它无权重或不需要初始化的模块（如 GELU、Dropout、ModuleList）自动跳过。
        """
        for m in self.modules():  # 遍历自身及所有子模块
            # —— 对一维卷积层：用 Kaiming Normal（适配 ReLU/GELU 一类激活），偏置置 0
            if isinstance(m, nn.Conv1d):
                # 按 fan_in（输入通道×卷积核长度）设置权重的正态分布方差，保持前向/反向方差稳定
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                # 偏置项若存在，置为 0（常见且稳健的做法）
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # —— 对全连接层：用 Kaiming Uniform（均匀分布版 He 初始化）
            if isinstance(m, nn.Linear):
                # a=√5 用于计算 gain（历史默认与 LeakyReLU 增益等价，实践上也常见）
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    # 线性层偏置使用 U(-1/√fan_in, 1/√fan_in)，与 PyTorch Linear 的默认策略一致
                    import math as _m
                    fan_in = m.weight.shape[1]                 # 输入特征数
                    bound = 1 / _m.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2: raise ValueError(f"期望 [B, N]，实际 {x.shape}")
        B, N = x.shape
        if N != self.seq_len: raise ValueError(f"序列长度不匹配：初始化 {self.seq_len}，输入 {N}")
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            y = spec(x); y = y + pw(x); x = self.act(y); x = self.dropout(x)
        x = x.mean(-1)
        return self.readout(x)
