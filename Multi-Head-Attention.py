import torch
import torch.nn as nn
import torch.nn.function as F
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, d_model):
    super().__init__()

    assert d_model % num_heads == 0

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)

    self.W_o = nn.Linear(d_model, d_model)

  def forward(self, x, mask = None):
    B, T, D = x.shape

    Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1))

    scores = scores / math.sqrt(self.d_k)

    if mask is not None:
      scores = scores.masked_fill(mask == 0, float('-inf))

      attn = F.softmax(scores, dim=-1)

      out = torch.matmul(attn, V)

      out = out.transpose(1, 2).comtiguous().view(B, T, D)

      out = self.W_o(out)

      return out

















