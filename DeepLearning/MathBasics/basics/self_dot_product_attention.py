import torch 
import math


def dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    '''
    实现 Scaled Dot Product Attention 机制: 

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) x V

    参数说明: 
        Q: 查询向量矩阵 (batch_size, seq_len_q, d_k)
           每个 token 用于发起“我要关注谁”的请求
        K: 关键向量矩阵 (batch_size, seq_len_k, d_k)
           每个 token 提供“我是谁”的标识, 被 query 扫描
        V: 值向量矩阵 (batch_size, seq_len_k, d_v)
           每个 token 实际包含的内容信息

    计算步骤: 
        1. 对每个 Query 和每个 Key 做点积, 得到相似度得分 QK^T
        2. 将分数除以 sqrt(d_k), 防止数值过大导致 softmax 梯度过小
        3. 对每一行做 softmax, 得到 attention 权重(每个 query 如何加权每个 key) 
        4. 将权重乘以 V, 计算加权平均, 得到每个 query 的最终输出表示

    返回值: 
        输出的 attention 表示, shape = (batch_size, seq_len_q, d_v)

    注意: 
        - 默认 Q, K, V 都是 batch 化后的 3D 张量
        - 若未 batch, 可在输入前增加维度: unsqueeze(0)
    '''
    d_k = K.size(-1)  # 最后一个维度, 也就是 d_k

    # 第一步: 点积得到 attention score
    scores = compute_attention_scores(Q, K)

    # 第二步: 缩放
    scores = scores / math.sqrt(d_k)

    # 第三步: softmax 得到注意力权重
    attention_weights = torch.softmax(scores, dim=-1)

    # 第四步: 加权求和 V
    output = torch.matmul(attention_weights, V)  # (batch, seq_len_q, d_v)

    return output



def compute_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    计算 Attention 中的 QK^T 得分矩阵(也叫 attention score matrix) 

    参数: 
        Q: 查询向量矩阵(Query) , shape: (batch_size, seq_len_q, d)
            - seq_len_q 表示 query 序列的长度(例如句子有几个 token) 
            - d 表示每个 token 被编码后的向量维度(通常是 512) 
            - token_vector = [x1, x2, x3, ..., x512]
        
        K: 关键向量矩阵(Key) , shape: (batch_size, seq_len_k, d)
            - seq_len_k 表示 context 序列的长度(可以和 L_q 相同, 也可以不同) 
            - d 是和 Q 一样的维度, 用于点积匹配

    返回: 
        scores: 注意力得分矩阵, shape: (L_q, L_k)
            - 表示每个 Query token 对每个 Key token 的“相似度”打分(点积) 
            - 每一行表示一个 Query 与所有 Key 的 dot product
              → 通常接下来会对每一行做 softmax, 得到注意力权重

    举例说明: 
        若句子中有 5 个词, 每个词被编码为一个 512 维向量, 
        那么 Q 和 K 的 shape 都是 (5, 512)
        结果 scores = Q @ K.T 的 shape 就是 (5, 5), 表示每个词对其他词的注意力得分
    """
    K_T = K.transpose(-2, -1)
    return torch.matmul(Q, K_T)





# 模拟高维attention_score计算
batch_size = 1       #样本数（句子）
seq_len = 4          #每个样本多少个token（词语）
d_model = 8          #每个token是8维向量(512 in real world)

Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

print("============================Tensors===========================")
print("Q shape:", Q.shape)
print("K shape:", K.shape)


print("============================scores===========================")
scores = compute_attention_scores(Q, K)
print("score shape: ", scores.shape)    # (1 X 4 x 8) @ (1 X 4 X 8).transpose() = (1 x 4 x 8) @ (1 x 8 x 4) = (1 x 4 x 4)


output = dot_product_attention(Q, K, V)
print()
print("==========================attention==========================")
print("V shape:", V.shape)
print("Output shape:", output.shape)
# print("Output tensor:\n", output)