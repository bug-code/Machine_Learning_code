from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
 
data = load_breast_cancer()
X = data.data
y = data.target
 
np.unique(y) #查看label都由哪些分类
plt.scatter(X[:,0], X[:,1],c=y)
plt.show()
 
 
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
# kernel = ["linear","poly","rbf", "sigmod"]
# for kernel in kernel:
#     time0 = time()
#     clf = SVC(kernel = kernel, gamma = "auto", cache_size=5000).fit(Xtrain,Ytrain) #cache_size default=200 默认使用200MB内存
#     print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
#     print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
 
#poly多项式核函数太浪费时间了，不能用
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
kernel = ["linear","rbf", "sigmoid"]
for kernel in kernel:
    time0 = time()
    clf = SVC(kernel = kernel, gamma = "auto", cache_size=5000).fit(Xtrain,Ytrain) #cache_size default=200 默认使用200MB内存
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
 
 
#到此应该可以确定乳腺癌数据集应该是一个线性可分的数据集
 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
kernel = ["linear","poly","rbf", "sigmoid"]
for kernel in kernel:
    time0 = time()
    clf = SVC(kernel = kernel, 
              degree = 1, #degree默认值是3，所以poly核函数跑的非常慢，
              gamma = "auto", 
              cache_size=5000).fit(Xtrain,Ytrain) #cache_size default=200 默认使用200MB内存
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
 
# # rbf 表现不应该这么差的，应该可以调整参数，对数据进行预处理，提高其准确性
 
import pandas as pd
data = pd.DataFrame(X)
 
data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
 
#数据存在问题：
# （1）查看数据均值mean 和 std 发现，数据量纲不统一
# (2) 数据的分布是偏态的 
# 因此我们需要对数据进行标准化
from sklearn.preprocessing import StandardScaler #让数据服从标准分布的标准化
X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
 
#数据处理之后，再跑一遍
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
kernel = ["linear","rbf", "sigmoid"]
for kernel in kernel:
    time0 = time()
    clf = SVC(kernel = kernel, 
              degree = 1, #degree默认值是3，所以poly核函数跑的非常慢，
              gamma = "auto", 
              cache_size=5000).fit(Xtrain,Ytrain) #cache_size default=200 默认使用200MB内存
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
   
 
# fine tune rbf parameter gamma
score = []
gamma_range = np.logspace(-10,1,50)
for i in gamma_range:
    clf = SVC(kernel="rbf", gamma=i, cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest, Ytest))
    
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range, score)
plt.show()
#其实到这里可以看到rbf的精度已经跟linear一致了，但是rbf核函数算的明显要快很多，所以用SVM大部分情况下，都使用rbf核函数
 
plt.plot(range(50), np.logspace(-10,1,50))
plt.show()
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
time0 = time()
gamma_range = np.logspace(-10, 1, 20)
coef0_range = np.linspace(0,5, 10)
param_grid = dict(gamma = gamma_range, coef0 = coef0_range)
cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state=420)
grid = GridSearchCV(SVC(kernel="poly", degree=1, cache_size=5000),param_grid=param_grid, cv=cv)
grid.fit(X,y)
print("The best parameters are %s with score %0.5f" % (grid.best_params_,grid.best_score_))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
 
#软间隔系数C 的调教， c默认1，必须是一个大于0 的数字