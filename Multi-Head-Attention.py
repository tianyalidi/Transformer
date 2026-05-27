import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0   # 每个 head 的维度必须整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 的线性映射
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出映射
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """

        B, T, D = x.shape

        # 1️⃣ 线性变换得到 Q K V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2️⃣ 拆分多头
        # (B, T, D) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3️⃣ attention score
        # Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # scaling（非常关键！！）
        scores = scores / math.sqrt(self.d_k)

        # 4️⃣ mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5️⃣ softmax
        attn = F.softmax(scores, dim=-1)

        # 6️⃣ 加权求和
        out = torch.matmul(attn, V)

        # 7️⃣ 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 8️⃣ 输出映射
        out = self.W_o(out)

        return out
