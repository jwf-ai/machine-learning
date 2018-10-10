# encoding: utf-8
import numpy as np
import math

class LLGC:

    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.inst_num = 0
        self.cls_num = 0
        self.NO_LABEL_ID = -1
        self.factor = 1.0
        self.sigma = 1.0
        self.iterations = 0

    # def fit(self, X, y):
    #     """
    #     训练模型
    #     :param X:训练集
    #     :param y:标签，如果无标签，以-1标识,类别从0开始标注
    #     :return:
    #     """
    #     self.inst_num = len(X)
    #     self.cls_num = len(set(y))-1 # 减去无标签（-1）
    #     X = np.array(X)
    #     y = np.array(y)
    #
    #     matrix_Y = self._cal_matrix_Y(y)
    #     matrix_W = self._cal_matrix_W(X)
    #     matrix_D = self._cal_matrix_D(matrix_W)
    #     matrix_S = self._cal_matrix_S(1,matrix_D,matrix_W)
    #     matrix_F = self._cal_matrix_F(matrix_S,matrix_Y)
    #
    #
    #     print "Y----------------"
    #     print matrix_Y
    #     print "W----------------"
    #     print matrix_W
    #     print "D----------------"
    #     print matrix_D
    #     print "S----------------"
    #     print matrix_S
    #
    #
    # def predict(self, X):
    #     pass

    def predict_labels(self, train_X, train_y, test_X):
        """
        预测测试集标签
        :param train_X: 训练集数据
        :param train_y: 训练集标签，如果无标签，以-1标识,类别从0按顺序标注
        :param test_X:  测试集数据
        :return: 测试集标签
        """
        test_y = [-1] * len(test_X) # 默认标签-1
        train_y.extend(test_y)
        train_X.extend(test_X)
        self.inst_num = len(train_X)
        self.cls_num = len(set(train_y))-1

        X = np.array(train_X)
        y = np.array(train_y)

        matrix_Y = self._cal_matrix_Y(y)
        matrix_W = self._cal_matrix_W(X)
        matrix_D = self._cal_matrix_D(matrix_W)
        matrix_S = self._cal_matrix_S(1, matrix_D, matrix_W)
        matrix_F = self._cal_matrix_F(matrix_S, matrix_Y)

        matrix_F_pre = matrix_F[len(train_X)-len(test_X):len(train_X)]
        labels = self._get_lables(matrix_F_pre)

        return labels

    def _cal_matrix_Y(self, y):
        """
        记录标签矩阵Y
        :param inst_num: 样本个数
        :param cls_num: 标签个数
        :return:
        """
        matrix_Y = np.zeros((self.inst_num,self.cls_num),np.float64)
        for i in xrange(self.inst_num):
            # 如果第i个样本有标签
            if y[i] != self.NO_LABEL_ID:
                matrix_Y[i][y[i]] = 1.0 #第i个样本的第j（标签）列置为1
        return matrix_Y

    def _cal_Distance(self, p1, p2):
        assert len(p1) == len(p2),"can not calculate distance because two points havs different dimension"
        sum = 0
        for i in xrange(len(p1)):
            sum += (p1[i]-p2[i])*(p1[i]-p2[i])

        return math.sqrt(sum)

    def _cal_matrix_W(self, X):
        """
        计算关联矩阵W
        :param X: 数据集
        :return:
        """
        matrix_W = np.zeros((self.inst_num,self.inst_num),np.float64)
        for i in xrange(self.inst_num):
            for j in xrange(self.inst_num):
                d = 0
                if i != j:
                    d = self._cal_Distance(X[i],X[j])
                    d = math.exp(-math.pow(d,2)/(2*self.sigma*self.sigma*self.factor))
                matrix_W[i][j] = d
        return matrix_W

    def _cal_matrix_D(self, matrix_W):
        """
        计算对角矩阵D
        :param matrix_W: 关联矩阵W
        :return:
        """
        matrix_D = np.zeros((self.inst_num, self.inst_num),np.float64)
        for i in xrange(self.inst_num):
            sum = 0
            for j in xrange(self.inst_num):
                sum += matrix_W[i][j]
            matrix_D[i][i] = sum
        return matrix_D

    def _cal_matrix_S(self, weighting_method, matrix_D, matrix_W):
        """
        计算传播矩阵
        :param weighting_method:
        :param matrix_D:
        :param matrix_W:
        :return:
        """
        if weighting_method == 1:
            # D^-1/2
            matrix_D_new = np.linalg.inv(np.sqrt(matrix_D))
            # S = D^-1/2 * W * D^-1/2
            matrix_S =np.dot(np.dot(matrix_D_new, matrix_W), matrix_D_new)
        elif weighting_method == 2:
            # P = D^-1 * W
            matrix_S = np.dot(np.linalg.inv(matrix_D), matrix_W)
        elif weighting_method == 3:
            # P^T = (D^-1 * W)^T
            matrix_S = np.dot((np.linalg.inv(matrix_D), matrix_W)).transpose()
        else:
            raise Exception("Unknow weighting method "+str(weighting_method))
        return matrix_S

    def _cal_matrix_F(self, matrix_S, matrix_Y):
        """
        收敛矩阵F
        :param matrix_S:
        :param matrix_Y:
        :return:
        """
        # matrix I
        matrix_F = np.identity(len(matrix_S), dtype=np.float64)

        # I - alpha * S
        matrix_F = matrix_F - np.dot(self.alpha, matrix_S)

        # 循环直到收敛？
        temp = matrix_F.copy()
        for i in xrange(self.iterations):
            matrix_F = np.dot(matrix_F,temp)

        # (I - alpha * S)^-1 * Y
        matrix_F = np.dot(np.linalg.inv(matrix_F), matrix_Y)

        return matrix_F

    def _get_lables(self, matrix_F_pre):
        """
        根据测试集对应的收敛矩阵，获得预测标签
        :param matrix_F_pre:
        :return:
        """
        labels = []
        for i in xrange(len(matrix_F_pre)):
            t = matrix_F_pre[i].tolist()
            label = t.index(max(t))
            labels.append(label)
        return labels

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import random

    iris = load_iris()

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i in xrange(3):
        indexs = range(i*50, i*50+50)
        test_index = random.sample(indexs, 10)
        test_X.extend(list(iris.data[x] for x in test_index))
        test_y.extend(list(iris.target[x] for x in test_index))

        train_index = list(x for x in indexs if x not in test_index)
        train_no_label_index = random.sample(train_index, 30)
        train_labeled_index = list(x for x in train_index if x not in train_no_label_index)

        train_X.extend(list(iris.data[x] for x in train_labeled_index))
        train_X.extend(list(iris.data[x] for x in train_no_label_index))
        train_y.extend(list(iris.target[x] for x in train_labeled_index))
        train_y.extend([-1]*30)

    import pandas as pd

    pd.DataFrame(train_X).to_csv("train_X.csv",index=False,header=False)
    pd.DataFrame(train_y).to_csv("train_y.csv", index=False, header=False)
    pd.DataFrame(test_X).to_csv("test_X.csv", index=False, header=False)
    pd.DataFrame(test_y).to_csv("test_y.csv", index=False, header=False)

    model = LLGC()
    pre_labels = model.predict_labels(train_X,train_y,test_X)

    for i in range(len(test_y)):
        print pre_labels[i],test_y[i]
    print "-----------------------------"

    from sklearn.semi_supervised import LabelPropagation
    m = LabelPropagation()
    m.fit(train_X,train_y)
    ppre_labels = list(x for x in m.predict(test_X))
    for i in range(len(test_y)):
        print ppre_labels[i],test_y[i]












