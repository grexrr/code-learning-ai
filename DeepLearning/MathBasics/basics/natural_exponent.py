import numpy as np
import matplotlib.pyplot as plt


# 自然对数图像
x = np.linspace(-2, 2, 400)
y1 = np.log(x)
y2 = np.exp(x)

fig, ax = plt.subplots()

ax.plot(x, y1, label= 'y = ln(x)', color='r')
ax.plot(x, y2, label= 'y = e^x', color='b')

ax.set_title('Natural Logarithm Function')
ax.set_xlabel('x')
ax.set_ylabel('y')

# ax.spines['left'].set_position('zero')  # 将左侧 y 轴移到 x=0
# ax.spines['right'].set_color('none')    # 隐藏右侧 y 轴
# ax.spines['top'].set_color('none')      # 隐藏顶部 x 轴
# ax.spines['bottom'].set_position('zero') # 将底部 x 轴移到 y=0


ax.grid(True)
ax.legend()

plt.show()