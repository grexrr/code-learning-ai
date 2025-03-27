import numpy as np
import matplotlib.pyplot as plt

"""
Softmax 函数介绍：
------------------
Softmax 是一种归一化函数,常用于分类模型的输出阶段,特别是神经网络的最后一层。

主要作用：
1. 将模型输出的一组任意实数(logits)变换成一组 0~1 的数,表示为“概率分布”。
2. 所有输出值之和为1,且每个值都 > 0,便于进行分类判断。
3. 常与交叉熵损失函数(Cross Entropy)一起使用,使模型输出的概率分布尽量接近真实标签(通常是one-hot的概率分布)。

数学定义：
softmax_i = exp(z_i) / sum(exp(z)),其中 z 是一组实数输入(比如神经网络输出)

数值稳定性技巧：
- 实际计算中,为防止 exp 溢出(尤其 z_i 很大时),我们通常减去 z 的最大值：
  softmax_i = exp(z_i - max(z)) / sum(exp(z - max(z)))

输出解释：
- softmax 的结果可以理解为模型对每个类别的“置信度”或“概率”
- 如果模型输出 logits=[2.3, 0.5, -1.2],经过 softmax 后可能是 [0.82, 0.15, 0.03]
  → 意思是“模型认为第1类的概率是82%”
"""

def softmax(X: np.ndarray) -> np.ndarray:
    """
    Softmax函数实现

    参数:
        X: 输入数组, 可以是一维或二维的 numpy 数组
           - 一维表示单个样本的logits
           - 二维表示多个样本, 每行一个样本的logits

    返回:
        归一化后的概率分布 (和为1,每个值都 > 0)

    实现原理:
    1. 公式: softmax(x)_i = exp(x_i) / sum(exp(x))
    2. 为了防止数值溢出(exp爆炸),常减去最大值:
       softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    3. 所有输出都为正,且总和为1,可视为“近似概率分布”
    """
    e_x = np.exp(X - X.max())  # 减去最大值,防止数值溢出
    return e_x / e_x.sum()     # 归一化成概率分布,和为1


# 示例: 对 [1, 2, 3] 进行 softmax 变换
X = np.array([[1, 2, 3]])
eX = np.exp(X - X.max())       # 手动计算 softmax 的分子
d = eX.sum()                   # 计算 softmax 的分母

# 验证 softmax 输出是概率分布(和为1)
print((eX / d).sum())          # 手动验证：应该接近 1
print(softmax(X).sum())        # 使用函数验证：也应该接近 1


# # 图示：Softmax 对一组连续数的变换效果
# x = np.linspace(-2, 2, 5)
# y = softmax(x)      # 虽然不是一组分类 logits,但可以看 softmax 如何压缩连续值

# fig, ax = plt.subplots()

# ax.plot(x, y, 'o', label='y = softmax(x)', color='black', linestyle='None')

# ax.set_title('Softmax Output Graph')
# ax.set_xlabel('x')
# ax.set_ylabel('y = softmax(x)')

# # 调整坐标轴位置,增加视觉美观
# # ax.spines['left'].set_position('zero')
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# # ax.spines['bottom'].set_position('zero')

# ax.grid(True)
# ax.legend()

# plt.show()


"""
Pytorch Version Below
"""
print()
print("#################Pytorch###############")



import torch
import torch.nn as nn

X = [1, 2, 3]
softmax = nn.Softmax(dim=0)
sigmaT = softmax(torch.tensor(X, dtype=torch.float32))
print(sigmaT)