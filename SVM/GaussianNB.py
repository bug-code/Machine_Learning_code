from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn import datasets
# from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score

##乳腺癌数据集
X,y=datasets.load_breast_cancer(return_X_y=True)
#将数据分为训练集和测试集
# x_train ,x_test,y_train ,y_test = train_test_split(X,y,test_size = 0.3)
# #让数据呈现标准正态分布
# sc = StandardScaler()
# x_train= sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# breast_cancer = datasets.load_breast_cancer()

#使用高斯贝叶斯算法进行分类
gnb = GaussianNB()
#使用训练集进行训练
# gnb.fit(x_train, y_train)
# # #预测训练集
# # predict_y_train_gnb=gnb.predict(x_train)
# # infos_gnb = "测试集准确率：" + str(metrics.accuracy_score(predict_y_train_gnb, y_test))
# #预测测试集
# predict_y_test_gnb = gnb.predict(x_test)
# # infos_gnb = "训练集准确率：" + str(metrics.accuracy_score(predict_y_test_gnb, y_train))
# # print( infos_gnb + " " + infos_gnb)
# right = 0
# for i in range(len(y_test)):
#     if predict_y_test_gnb[i]==y_test[i]:
#         right+=1
# print(right/len(y_test))
scores=cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())