import torch
import math

def compute_attention_scores(Q, K):
    # Q: (B, H, L_q, D), K: (B, H, L_k, D)
    # transpose 最后两个维度，准备做 Q x K^T
    K_T = K.transpose(-2, -1)  # → (B, H, D, L_k)
    return torch.matmul(Q, K_T)  # → (B, H, L_q, L_k)

def dot_product_attention(Q, K, V, mask=None):
    """
    标准 Scaled Dot Product Attention 计算(支持多头): 
    1. scores = QK^T / sqrt(d_k)
    2. softmax(scores)
    3. 加权求和: softmax x V
    """
    d_k = Q.size(-1)  # 每个 head 的 depth
    scores = compute_attention_scores(Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # 屏蔽掉无效位置

    weights = torch.softmax(scores, dim=-1)  # 注意力权重 (B, H, L_q, L_k)
    return torch.matmul(weights, V)          # 输出 shape: (B, H, L_q, D)

def split_heads(x, num_heads):
    """
    把输入 (B, L, D) 拆成多头表示 → (B, num_heads, L, depth)

    思路: 
    1. 原始向量是 D 维(例如 D=8)，要拆成 H 个头(例如 H=2)
    2. 每个头的维度 = D / H = 4
    3. reshape 先把 D 拆成 (H, depth)，变成 (B, L, H, depth)
    4. transpose: 交换 H 和 L → (B, H, L, depth)

    📌 为什么交换？
    - 因为我们希望 **每个头能独立处理一整句(seq_len)**
    - 所以要把 “每个 token 的多个头向量” 变成 “每个头的一整组 token”
    """
    B, L, D = x.shape
    depth = D // num_heads
    x = x.reshape(B, L, num_heads, depth)    # (B, L, H, D') ← 把 d_model 拆成多个头
    return x.permute(0, 2, 1, 3)             # (B, H, L, D') ← 每个头自己看一整句

def combine_heads(x):
    """
    将多个头的输出重新拼接回原始维度 (B, L, D_model)

    思路: 
    1. 原始多头输出是 (B, H, L, D)
    2. 我们想要 (B, L, H * D)
    3. 所以先 permute，把 L 放回第二维，再 reshape 拼接头
    """
    B, H, L, D = x.shape
    return x.permute(0, 2, 1, 3).reshape(B, L, H * D)

# ========================= 测试 =========================

# 假设: 1 个 batch，句子长度 4，每个 token 是 8 维向量
batch_size, seq_len, d_model, num_heads = 1, 4, 8, 2
depth = d_model // num_heads

x = torch.randn(batch_size, seq_len, d_model)  # 模拟输入句子 (B, L, D)

# 拆分多头: Q, K, V 都从 x 拷贝
Q = split_heads(x, num_heads)  # → (B, H, L, D')
K = split_heads(x, num_heads)
V = split_heads(x, num_heads)

attn_output = dot_product_attention(Q, K, V)  # → (B, H, L, D')
output = combine_heads(attn_output)          # → (B, L, D_model)

# ========================= 打印维度 =========================
print("输入 shape:", x.shape)                     # (1, 4, 8)
print("拆多头后 Q shape:", Q.shape)              # (1, 2, 4, 4)
print("Attention 输出 shape:", attn_output.shape) # (1, 2, 4, 4)
print("拼接后输出 shape:", output.shape)          # (1, 4, 8)
