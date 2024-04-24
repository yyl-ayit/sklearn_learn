import matplotlib.pyplot as plt
import numpy as np

class Xianxing:

    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        read1=Xianxing.get_data()
        self.x, self.y = zip(*read1)

    def get_error_value(self, w, b):
        total_error=0
        for i in range(len(self.x)):
            x1 = self.x[i]
            y1 = self.y[i]
            total_error += (y1-(w*x1+b))**2
        return total_error/len(self.x)

    @classmethod
    def get_data(cls):
        with open(r"D:\桌面\日常练习\深度学习\参数\随机线性数.txt", 'r', encoding='utf-8') as f:
            read1 = f.read()
        return eval(read1)
    def m_b_draw(self, m, b):
        fig, ax = plt.subplots()
        xx = np.linspace(0, 100, 100)
        yy = m * xx + b

        ax.plot(xx, yy)
        ax.scatter(self.x, self.y, color="black")
        ax.set_title("线性随机数")
        ax.set_xlabel('X')
        ax.set_ylabel("Y")

        plt.show()
    def up_w_b(self, w_current, b_current, learning_rate):
        b_gradient = 0
        w_gradient = 0
        for i in range(len(self.x)):
            xx=self.x[i]
            yy=self.y[i]
            b_gradient += -(2 * (yy - w_current * xx - b_current))
            w_gradient += -(2 * xx * (yy - w_current * xx - b_current))
        new_b = b_current - (learning_rate * b_gradient / len(self.x))
        new_w = w_current - (learning_rate * w_gradient / len(self.x))
        return new_w, new_b
    def optimize_w_b(self, w, b, learning_rate, num_iterations):
        for i in range(num_iterations):
            w, b=self.up_w_b(w, b, learning_rate)
        return w, b

    def __str__(self):
        return "简单线性分析"


if __name__ == '__main__':
    P=Xianxing()
    init_w = 0
    init_b = 0
    rate=0.0001
    num=10000
    ww, bb=P.optimize_w_b(init_w, init_b, rate, num)
    # print("parameter w and b:", w, b)
    # print("Error:", P.get_error_value(w, b))
    # P.m_b_draw(w, b)
    #print(P.get_data())
    P.m_b_draw(0,0)



