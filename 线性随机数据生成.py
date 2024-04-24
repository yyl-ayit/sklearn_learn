import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机数种子，保证结果可重现（可选操作）
#np.random.seed(1)

# 生成x坐标的随机数

x = np.random.uniform(0, 100, 500)

# 生成y坐标的随机数
y = np.random.uniform(0, 100, 500)
y.sort()
x.sort()
# flg,ax = plt.subplots()
#
# ax.scatter(x, y, color="black")
# ax.set_title("线性随机数")
# ax.set_xlabel('X')
# ax.set_ylabel("Y")
# # 打印生成的随机数对
random_points = list(zip(x, y))
with open(r"D:\桌面\日常练习\深度学习\参数\随机线性数.txt",'w',encoding='utf-8') as f:
    f.write(str(random_points))
# plt.show()
# print(random_points)