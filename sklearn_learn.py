import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    # 特征数据
    data_x = breast_cancer.data
    # 对应标签
    data_y = breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(X_train)
    x_test = transfer.transform(X_test)
    estimator = KNeighborsClassifier()
    param_grid = {'n_neighbors': [i for i in range(1, 21)]}

    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
    estimator.fit(x_train, y_train)
    print('estimator.best_score_---', estimator.best_score_)
    print('estimator.best_estimator_---', estimator.best_estimator_)
    print('estimator.best_params_---', estimator.best_params_)

    myret = pd.DataFrame(estimator.cv_results_)
    myret.to_csv(path_or_buf='./mygridsearchcv.csv')
