from sklearn.datasets import load_iris
import random

iris = load_iris()

train_X_labeled = []
train_X_unlabeled = []
train_y_labeled = []
train_y_unlabeled = []
test_X = []
test_y = []
for i in xrange(3):
    indexs = range(i * 50, i * 50 + 50)
    test_index = random.sample(indexs, 10)
    test_X.extend(list(iris.data[x] for x in test_index))
    test_y.extend(list(iris.target[x] for x in test_index))

    train_index = list(x for x in indexs if x not in test_index)
    train_no_label_index = random.sample(train_index, 30)
    train_labeled_index = list(x for x in train_index if x not in train_no_label_index)

    train_X_labeled.extend(list(iris.data[x] for x in train_labeled_index))
    train_X_unlabeled.extend(list(iris.data[x] for x in train_no_label_index))
    train_y_labeled.extend(list(iris.target[x] for x in train_labeled_index))
    train_y_unlabeled.extend([-1] * 30)

import pandas as pd

pd.DataFrame(train_X_labeled).to_csv("train_X_labeled.csv", index=False)
pd.DataFrame(train_X_unlabeled).to_csv("train_X_unlabeled.csv", index=False)
pd.DataFrame(train_y_labeled).to_csv("train_y_labeled.csv", index=False)
pd.DataFrame(train_y_unlabeled).to_csv("train_y_unlabeled.csv", index=False)
pd.DataFrame(test_X).to_csv("test_X.csv", index=False)
pd.DataFrame(test_y).to_csv("test_y.csv", index=False)