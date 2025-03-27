import numpy as np

def softmax(X: np.ndarray) -> np.ndarray:
    e_x = np.exp(X - X.max())  # 减去最大值,避免数值爆炸
    return e_x / e_x.sum()     # 归一化为概率分布

def entropy(X: np.ndarray, base: float = np.e) -> np.ndarray:
    """
    熵(Entropy)函数:衡量一个概率分布的不确定性(混乱程度)

    参数:
        X: 一个概率分布数组,要求值 ∈ [0,1],且总和为1
        base: 对数的底,默认为自然对数(e);可以设为2 表示比特信息量

    返回:
        一个浮点数,表示该分布的信息熵。熵越大,分布越“均匀”; 熵越小,分布越“确定”。

    原理公式:
        entropy(X) = -sum( p_i * log(p_i) )
    
    举例:
        若 X = [1, 0, 0] → 熵 = 0(完全确定,只有一个类)
        若 X = [1/3, 1/3, 1/3] → 熵最大(完全不确定,每类一样可能)
    """

    X = np.clip(X, 1e-12, 1)  # 避免 log(0),防止数值错误
    return -np.sum(X * np.log(X) / np.log(base))  # 熵公式



# ========================= 示例1:均匀随机分布 =========================
# 生成一个 1x8 的随机浮点数数组,范围在 [-2.0, 2.0)
X = np.random.uniform(-2.0, 2.0, size=(1, 8))
softmax_X_rand = softmax(X)            # 转换为概率分布
print("随机均匀 softmax 概率分布的熵 =", entropy(softmax_X_rand))

# ========================= 示例2:正态分布随机数 =========================
# 生成一个 1x8 的标准正态分布随机数

X_norm = np.random.randn(1, 8)
softmax_X_norm = softmax(X_norm)
print("正态分布 softmax 概率分布的熵 =", entropy(softmax_X_norm))


def crossEntropy(P: np.ndarray, Q: np.ndarray, base: float = np.e) -> np.ndarray:
    """
    交叉熵 (Cross Entropy) 函数: 
    ---------------------------------------
    用于衡量两个概率分布之间的差异, 通常用于分类问题中: 

    - P: 真实概率分布 (标签, ground truth) 
    - Q: 模型预测出来的概率分布 (通常来自 softmax) 

    数学公式: 
        H(P, Q) = -∑ P(x) * log(Q(x))
    
    如果 P 是 one-hot, 那么只有正确那一类会参与损失计算: 
        H = -log(Q[正确类])

    参数:
        P: numpy 数组, 表示真实标签的概率分布 (一般是 one-hot 向量) 
        Q: numpy 数组, 表示模型预测的概率分布 (softmax输出) 
        base: 对数的底数 (默认为自然对数 e); 可以设为2表示以 bit 为单位

    返回:
        float 类型的交叉熵损失值 (越小越好) 

    数值稳定性处理:
        使用 np.clip() 将所有输入值限制在 [1e-12, 1], 防止 log(0) 导致 NaN 或 -inf
    """
    P = np.clip(P, 1e-12, 1)  # 保证标签概率不为0, 避免 log(0)
    Q = np.clip(Q, 1e-12, 1)  # 同样防止预测概率为0
    return -np.sum(P * np.log(Q) / np.log(base))  # 交叉熵计算公式


# ====================== 示例1：模型预测得很好 ============================
P = np.array([0, 0, 1, 0])  # 真实标签：第三类
Q_good = np.array([0.05, 0.1, 0.8, 0.05])  # 模型很自信地预测对了
loss_good = crossEntropy(P, Q_good)
print("预测正确且自信时的交叉熵 =", loss_good)

# ====================== 示例2：模型预测得一般 ============================
Q_okay = np.array([0.2, 0.3, 0.4, 0.1])  # 模型不太确定
loss_okay = crossEntropy(P, Q_okay)
print("预测一般的交叉熵 =", loss_okay)

# ====================== 示例3：模型预测错得很自信 ============================
Q_bad = np.array([0.98, 0.01, 0.005, 0.005])  # 模型强烈认为是第1类，错得离谱
loss_bad = crossEntropy(P, Q_bad)
print("预测错误且自信时的交叉熵 =", loss_bad)


# ==================== 示例3：模型完全平均分布 =======================
Q_equal = np.array([1/4 for _ in range(4)])  # 模型完全平均，不知道答案
loss_compare = crossEntropy(P, Q_equal)
print("交叉熵 =", loss_compare)