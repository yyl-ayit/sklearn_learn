import numpy as np
from sklearn import linear_model
import plotly.graph_objects as go

if __name__ == '__main__':
    # 随机产生实验数据
    data_x = [[np.random.normal(loc=0.0, scale=1.0, size=1)[0]] for i in range(1000)]
    data_y = np.random.normal(loc=0.0, scale=1.0, size=1000)
    # 更改参数，查看效果
    reg = linear_model.BayesianRidge()
    reg.fit(data_x, data_y)
    x = [sum(i) for i in data_x]
    y1 = reg.predict(data_x)
    fig = go.Figure(data=[
        go.Scatter(x=x, y=data_y, mode='markers', name='原点'),
        go.Scatter(x=x, y=y1, mode='markers', name='预测')
    ])
    fig.show()
