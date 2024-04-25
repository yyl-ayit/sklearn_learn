import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
if __name__ == '__main__':
    # 生成一个回归数据集
    data_x, data_y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
    kr = KernelRidge(kernel='rbf', gamma=0.1)

    # 使用网格搜索来寻找最佳的超参数组合
    param_grid = {'alpha': [1e0, 1e-1, 1e-2, 1e-3],
                  'gamma': [0.1, 0.01, 0.001, 0.0001]}

    kr_grid = GridSearchCV(kr, param_grid=param_grid, cv=5)
    kr_grid.fit(X_train, y_train)
    print("Best parameters: ", kr_grid.best_params_)
    print("Best score: ", kr_grid.best_score_)
    y_pred = kr_grid.predict(data_x)
    data_x = [sum(i) for i in data_x]
    fig = go.Figure(data=[
        go.Scatter(x=data_x, y=data_y, mode='markers', name='原始点集'),
        go.Scatter(x=data_x, y=y_pred, mode='lines', name='预测')
    ])
    fig.show()
