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

def lr_unary(data,target):

    # make the sixth feature to generate unary data
    data = list(x[5] for x in data)
    print(data,target)

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


        print("dw:",dw,"db:",db)


        if w==w_save and b==b_save:
            break
        else:
            w_save = w
            b_save = b
            i += 1


    plt.show()







if __name__ == "__main__":
    data, target = load_data()


    lr_unary(data,target)