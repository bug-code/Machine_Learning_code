import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn import datasets
from sklearn import metrics
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


##乳腺癌数据集
X,y=datasets.load_breast_cancer(return_X_y=True)
#让数据呈现标准正态分布
sc = StandardScaler()
X= sc.fit_transform(X)
#将数据分为训练集和测试集
x_train ,x_test,y_train ,y_test = train_test_split(X,y,test_size = 0.3)
#测试乳腺癌数据集适用的核函数
kernels = ["linear","rbf", "sigmoid"]
score = []
for kernel in kernels:
    clf = SVC(kernel = kernel, 
              degree = 1, #degree默认值是3，所以poly核函数跑的非常慢，
              gamma = "auto", 
              cache_size=5000).fit(x_train,y_train) #cache_size default=200 默认使用200MB内存
    score.append(clf.score(x_test, y_test))
#绘制相应核函数分类得分柱状图
plt.bar(range(len(score)), score,color='rgb',tick_label=kernels)  
plt.title('对应核函数得分柱状图') 
plt.legend()
plt.show()  
##使用SVM进行训练
#创建SVM类
SVC = SVC(kernel='linear')
gnb = GaussianNB()
tree = tree.DecisionTreeClassifier()
#使用训练集对SVM进行训练
SVC.fit(x_train,y_train)
#####################################################
scores_svc=cross_val_score(SVC, X, y, cv=10, scoring='accuracy')
scores_gnb=cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
scores_tree=cross_val_score(tree, X, y, cv=10, scoring='accuracy')
scores_means =[scores_svc.mean(),scores_gnb.mean(),scores_tree.mean()]
ALGs=['SVM','gnb','tree']
# print(scores_means) 
#绘制相应核函数分类得分柱状图
plt.bar(range(len(scores_means)), scores_means,color='rgb',tick_label=ALGs)  
plt.title('三种分类算法平均准确率') 
plt.legend()
plt.show()  
#画出实际值和预测值折线图
plt.plot(scores_svc, marker = 'o',color='red', label = 'SVM')
plt.plot(scores_gnb, marker = 'x', color='green',label = 'gnb')
plt.plot(scores_tree, marker = '*', color='blue',label = 'tree')
plt.xlabel('实验次数')
plt.ylabel('准确率')
plt.title('三种分类算法十次训练效果图')
plt.legend()
plt.show()

#####################################################
##随机投影降维
rp=random_projection.SparseRandomProjection(n_components=3,density=0.1,random_state=0)
X_projected=rp.fit_transform(X)
##结果标准化
X_projected=preprocessing.scale(X_projected)
#对整个数据集进行降维投影
plt.figure(1)
ax = plt.subplot(111, projection='3d')
ax.scatter(X_projected[:,0],X_projected[:,1],X_projected[:,2],c=y)

#可视化分类超平面
w = SVC.coef_ 
w=rp.fit_transform(w) 
b = SVC.intercept_
x = np.arange(-5,5,0.5)
y = np.arange(-5,5,0.5)
x, y = np.meshgrid(x, y)
z = (w[0,0]*x + w[0,1]*y + b) / (-w[0,2])
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)
plt.show()
