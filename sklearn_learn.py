import numpy as np
from sklearn import linear_model
import plotly.graph_objects as go

if __name__ == '__main__':
    # 随机产生实验数据
    data_x = [[np.random.randint(0, 1000, size=1)[0]] for i in range(1000)]
    data_y = np.random.randint(0, 1000, size=1000)

    # 更改参数，查看效果
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit(data_x, data_y)
    x = [i[0] for i in data_x]
    print(reg.coef_)
    fig = go.Figure(data=[
        go.Scatter(x=x, y=data_y, mode='markers', name='原点'),
        go.Scatter(x=x, y=x * reg.coef_ + reg.intercept_, mode='lines', name='岭回归')
    ])
    fig.show()
