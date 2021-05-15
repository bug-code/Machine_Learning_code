import csv
from datetime import datetime
import random
import numpy
import pandas
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
     背景：
            一 通过随机生成数据样本集来训练感知器网络
            二 感知器网络对实际样本数据的拟合程度与训练样本有关
                    1 训练样本的数量越多，感知器网络越能够逼近实际对应的线性超平面，但是并不是训练样本数量越多越好。
                      当训练样本数量足够多时，就已经能以很小的误差拟合真实的线性超平面，如果要进一步缩小误差，
                      所花费的代价将变得更大

                    2 训练样本的质量。感知器网络对实际的线性超平面拟合程度也与训练样本质量有关。
                      训练样本质量越好越能够拟合实际线性超平面。

                    3 学习率 eta 。感知器网络也与学习率有关。学习率越小，对实际线性超平面拟合的误差越小，但是其学习的
                      时间也越长。学习率越大，对实际线性超平面拟合的误差越大，但是其学习的时间越快。

                    4 训练次数。感知器网络的拟合程度和训练次数有关。训练次数本质上和训练样本数量的性质一样。在每次训练的
                      实际样本集中，实际对感知器网络产生影响的是那些感知器预测错误的样本。因为只有当预测错误了才能产生误差
                      对多感知器网络的权值进行调整，从而拟合实际对应的线性超平面。训练样本越多预测样本的数量也就越多，越能
                      成功拟合，在每次训练中都是对上一次出错的样本数据进行训练宠幸调整权值。
            三 补充
                    在该程序生成的数据集中，数据 X 和 Y 都是高精度浮点数 ， 由（X , Y）组成的点几乎可以认为是不重复的。
                    所以可以不用对数据集进行预处理。
'''


# 生成随机数据样本
def creat_csvfile():
    datestr = (datetime.now()).strftime("%d%b%Y-%H%M%S")

    file = 'data-' + datestr + '.csv'
    with open(file, 'w', newline="") as f:
        csv_write = csv.writer(f)
        csv_head = ['x', 'y', 'target']
        csv_write.writerow(csv_head)
    return file


def write_csvfile(fileName, data):
    with open(fileName, 'a+', newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerows(data)


def set_function(x, y):
    result = 0.5 * x + 1 - y
    return result


def produce_data(sumN):
    lists = []
    for N in range(0, sumN):
        lis = []
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        if set_function(x, y) > 0:
            lis.extend((x, y, 'true'))
        else:
            lis.extend((x, y, 'false'))
        lists.append(lis)

    return lists


def trans_attribute(arr):
    arrs = []
    for element in arr:
        if str(element) == 'True':
            arrs.append(1)
        else:
            arrs.append(-1)
    return arrs


# 感知器算法
'''
感知器算法：                    1   target>0    
            y = sign(target)= 
                              -1   target<0
            target = W1*X1+.....+Wn*Xn+W0
            sign(target)为激活函数，对输出节点的输出值进行二分判断
            W0为偏置因子
            W[i] i in range(1,n+1)  :为各节点到输出节点的权值
            将target改写为：
                            target = W0*X0+W1*X1+....+Wn*Xn=W*X
            其中将X0=1
            权值更行公式：Wj(k+1)=Wj(k)+eta*update*Xij
                        n 
感知器算法过程：输入训练样本数据通过感知器所得到的结果与训练样本数据的实际结果得到误差值Update,
通过Update不断修改感知器的输入权值，直到将感知器调整到
能够大部分拟合训练数据。           
'''


class proceptron(object):
    def __init__(self, eta, n_iter):
        self.eta = eta  # 学习率
        self.iter = n_iter  # 训练样本次数

    def fit(self, x, y):
        self.w = numpy.zeros(x.shape[1] + 1)
        self.errors = []
        for i in range(self.iter):
            error = 0
            for xi, target in zip(x, y):
                correction = self.eta * (target - self.prodict(xi))
                self.w[1:] += correction * xi
                self.w[0] += correction
                error += int(correction != 0.0)
            self.errors.append(error)

    def net_input(self, x):
        return numpy.dot(x, self.w[1:]) + self.w[0]

    def prodict(self, x):
        return numpy.where(self.net_input(x) >= 0, 1, -1)


# 生成样本文件
fileName = creat_csvfile()
lis = produce_data(200)
write_csvfile(fileName, lis)

# 读取数据文件，取出数据
read_file = pandas.read_csv(fileName, header=0)
target = read_file.loc[0:200, 'target'].values
target = trans_attribute(target)
x = read_file.iloc[0:200, [0, 1]].values
x_train = x[0:100]
target_train = target[0:100]
x_test = x[100:200]
target_test = target[100:200]

# 画出训练样本图像
for i in range(len(target_train)):
    if target_train[i] > 0:
        plt.scatter(x[i, 0], x[i, 1], color='black', marker='o')
    else:
        plt.scatter(x[i, 0], x[i, 1], color='red', marker='x')

plt.xlabel('x轴')
plt.ylabel('y轴')
plt.title('训练样本图像')
plt.legend(loc='upper left')
plt.show()

# 训练模型
procept = proceptron(eta=0.06, n_iter=10)
procept.fit(x_train, target_train)

# 画出错误次数折线图
plt.plot(range(1, len(procept.errors) + 1), procept.errors, marker='x')
plt.xlabel("次数")
plt.ylabel("错误次数")
plt.title('单次训练错误次数折线图')
plt.show()

# 画出所有训练样本图像与训练求得的超平面
x1 = 0
y1 = -1 / procept.w[2] * (procept.w[0] * 1 + procept.w[1] * x1)
x2 = 11
y2 = -1 / procept.w[2] * (procept.w[0] * 1 + procept.w[1] * x2)
for i in range(len(target)):
    if target[i] > 0:
        plt.scatter(x[i, 0], x[i, 1], color='black', marker='o')
    else:
        plt.scatter(x[i, 0], x[i, 1], color='red', marker='x')
plt.plot([x1, x2], [y1, y2], 'r')
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.legend(loc='upper left')
plt.show()

# 测试训练结果
correctN = 0
wrongN = 0
for i in range(100):
    consequet = procept.prodict(x_test[i]) * target_test[i]
    if consequet > 0:
        correctN += 1
    else:
        wrongN += 1
cor_and_wrong = []
cor_and_wrong.append(correctN)
cor_and_wrong.append(wrongN)
label = ["正确", "错误"]
plt.pie(cor_and_wrong, labels=label, autopct="%.2f%%")
plt.title('错误率饼图')
plt.legend(loc="best")
plt.show()