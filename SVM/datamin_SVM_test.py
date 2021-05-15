from sklearn import svm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
 
def train_func(func, features_remain, desc):
    #分类阶段：模型训练
    #抽取30%的数据作为测试集，其余作为训练集
    train, test = train_test_split(data, test_size=0.3)
    train_x = train[features_remain]
    train_y = train['diagnosis']
    test_x = test[features_remain]
    test_y = test['diagnosis']
 
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.transform(test_x)
 
    #创建SVM分类器
    if(func == "linear"):
        model = svm.LinearSVC()
    else:
        model = svm.SVC()
    #用训练集做训练
    model.fit(train_x, train_y)
    #用测试集做预测
 
    #分类阶段：模型评估
    predict_y = model.predict(test_x)
    infos1 = "测试集准确率：" + str(metrics.accuracy_score(predict_y, test_y))
 
    predict_yy = model.predict(train_x)
    infos2 = "训练集准确率：" + str(metrics.accuracy_score(predict_yy, train_y))
    print(desc + ":" + infos1 + " " + infos2)
 
 
#准备阶段：数据探索
data = pd.read_csv('./breast_cancer_data-master/data.csv')
#把所有的列都显示出来（在打印的时候）
pd.set_option('display.max_columns', None)
 
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
 
#准备阶段：数据清洗，id列没有用，删除该列
data.drop("id", axis=1, inplace=True)
# 将B良性替换为0，M恶性替换为1
data['diagnosis']=data['diagnosis'].map({'M': 1, 'B': 0})
 
#准备阶段：数据可视化
sns.countplot(data['diagnosis'], label='Count')
plt.show()
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
plt.show()
 
#分类阶段：特征选择
features_remain = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']
 
#分类阶段：模型训练+模型评估
train_func('svc', features_remain, 'svc_six')
train_func('svc', features_mean, 'svc_all')
train_func('linear', features_remain, 'linearsvc_six')
train_func('linear', features_mean, 'linearsvc_all')