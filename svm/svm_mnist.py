from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

max_per_class = 10
counter = {}
min_x_train = []
min_y_train = []

for i in range(len(y_train)):
    if y_train[i] not in counter:
        counter[y_train[i]] = 0
    if counter[y_train[i]] == max_per_class:
        continue
    min_x_train.append(x_train[i])
    min_y_train.append(y_train[i])
    counter[y_train[i]] += 1

x_train = np.array(min_x_train)
y_train = np.array(min_y_train)


x_train = [x.flatten() for x in x_train]
x_test = [x.flatten() for x in x_test]

param_grid={
    'C':[0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
    }
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [0.001],
#     'kernel': ['rbf']
# }
svc = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid, scoring='accuracy', verbose=10)

model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('Best params: ', model.best_params_)
print(acc)
