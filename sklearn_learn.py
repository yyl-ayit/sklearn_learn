import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go
if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


