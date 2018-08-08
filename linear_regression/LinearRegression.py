# encoding: utf-8
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """
    load data
    :return:
    """
    boston = load_boston()
    return boston.data, boston.target

def lr_unary():
    """
    unary logistic regression
    :param data:
    :param target:
    :return:
    """

    # make the sixth feature to generate unary data
    data, target = load_data()
    data = list(x[5] for x in data)

    # start linear fitting
    w=0
    b=0
    w_save=0
    b_save=0
    alpha = 0.01
    i=0
    while True:
        # the hypothesis function is f(x)=wx+b
        # calc the square loss
        loss = 0
        for (x,y) in zip(data,target):
            loss += 1/2*(w*x+b-y)**2
        loss = loss / len(data)

        # show the data
        if i % 100 == 0:
            plt.cla()
            plt.xlim((0, 10))
            plt.ylim((0, 60))
            plt.scatter(data, target, marker="x", c='b')
            plt.plot(np.arange(0,10),np.arange(0,10)*w+b,c='r')
            plt.xlabel("the sixth factor")
            plt.ylabel("Boston house-price")
            plt.title("iter:%s,loss:%s"%(i,loss))
            plt.pause(0.01)


        # calc the dw and db
        dw = 0
        for (x,y) in zip(data,target):
            dw += (w*x+b-y)*x
        dw = round(dw / len(data),4)

        db = 0
        for (x, y) in zip(data, target):
            db += w * x + b - y
        db = round(db / len(data),4)

        # update the w and b
        w -= alpha * dw
        b -= alpha * db

        if w==w_save and b==b_save:
            break
        else:
            w_save = w
            b_save = b
            i += 1


    plt.show()


def lr_multiple():
    """
    multiple linear regression
    :return:
    """

    def fun_sum(x, w):
        sum = 0
        for (xi, wi) in zip(x, w):
            sum += xi * wi
        return sum

    data, target = load_data()

    # para init
    w = np.zeros(len(data[0]))

    b = 0.0
    alpha = 0.000005

    for itr in range(3000000):

        # calc loss
        loss = 0.0
        for (X,y) in zip(data,target):
            loss += 1/2*(fun_sum(X,w)+b-y)**2
        loss = loss / len(data)
        if itr % 100 == 0:
            print("loss:",loss)

        # update dw and db
        dw = np.zeros(len(w))
        for j in range(len(dw)):
            dwj = 0
            for (X, y) in zip(data, target):
                dwj += (fun_sum(X, w) + b - y)*X[j]
            dw[j] = dwj / len(data)

        db = 0
        for (X, y) in zip(data, target):
            db += fun_sum(X, w) + b - y
        db = db / len(data)

        # update w and b
        for k in range(len(dw)):
            w[k] = w[k] - alpha * dw[k]
        b = b - alpha * db


def lr_multiple_mat():
    """
    using matrix to implement linear regression
    :return:
    """
    X, t = load_data()

    X = np.matrix(X)
    t = np.matrix(t)

    w = np.zeros((1, X.shape[1]))
    b = np.zeros((1,1))

    w_save = np.zeros((1, X.shape[1]))
    b_save = np.zeros((1, 1))

    alpha = 0.000005
    itr = 0

    for i in range(100000000):

        y = np.dot(X,w.T)+b
        # calc loss
        loss = 1/2 * np.dot((t.T-y).T,(t.T-y)) / X.shape[0]
        if itr % 10000 == 0:
            print("itr:",itr)
            print("loss:",loss[0,0])

        # calc dw,db
        dw = np.dot((y-t.T).T,X)/X.shape[0]
        db = np.dot(np.ones((1,X.shape[0])),(y-t.T))/X.shape[0]

        # update w, b
        w -= alpha * dw
        b -= alpha * db

        if (w_save==w).all() and (b_save==b).all():
            break
        else:
            w_save = w.copy()
            b_save = b.copy()
        itr += 1


    print("loss:", loss[0, 0])
    print("w:",[round(x,2) for x in w.tolist()[0]])
    print("b",b)


def lr_multiple_least_square():
    X, t = load_data()

    X = np.matrix(X)
    t = np.matrix(t).T

    print(X.shape,t.shape)

    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),t)
    print([round(x[0],2) for x in w.tolist()])















if __name__ == "__main__":



    # lr_unary()
    #
    # lr_multiple()
    lr_multiple_mat()
    lr_multiple_least_square()