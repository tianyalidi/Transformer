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

    Q = Q.view(B, T, self.nums_head, self.d_k)
    
