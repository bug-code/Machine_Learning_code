from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt
import numpy
import math
#设置中文编码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
'''
#频率和众数
'''
#生成测试数据
def create_dataset(sumN):
    datas = []
    for i in range(sumN):
        data = random.randrange(0,5,1)
        datas.append(data)
    return datas

#获取数据集的频率
def frequency(dataset,dict):
    for i in range(len(dataset)):
        for key in dict.keys():
            if int(key) == dataset[i] :
                dict[key] +=1
    sumN = sum(dict.values())
    for key  in dict.keys():
        dict[key] = dict[key]/sumN
    
#随机生成数据集
dataset = create_dataset(200)
#初始化数据集属性字典
dict = {'0':0,'1':0,'2':0,'3':0,'4':0}
#求出字典频率
frequency(dataset,dict)
#画出数据集频率饼图
label=["0","1","2","3","4"]
value = dict.values()
plt.pie(value,labels=label,autopct="%.2f%%")
plt.title('频率图')
plt.legend(loc="best")
plt.show()
'''

'''
#鸢尾花百分位数
'''
def percentile(datalist):
    attribute_list = []
    attribute_percentile = []
    for i in range(len(datalist)):
        if i==len(datalist)-1:
            attribute_list.append(datalist[i])
            attribute_percentile.append('100%')
        elif i<len(datalist)-1 and datalist[i]==datalist[i+1]  :
            pass
        else:
            attribute_list.append(datalist[i])
            attribute_percentile.append('{:.2%}'.format(i/len(datalist)))
    return attribute_list , attribute_percentile
iris = load_iris()
data = iris['data']
#萼片长度
SepalLength = sorted(data[:,0])
SepalLength_list , SepalLength_percentile = percentile(SepalLength)
#print(SepalLength_list,SepalLength_percentile)
#萼片宽度
SepalWidth = sorted(data[:,1])
SepalWidth_list , SepalWidth_percentile = percentile(SepalWidth)
#花瓣长度
PetalLength = sorted(data[:,2])
PetalLength_list , PetalLength_percentile = percentile(PetalLength)
#花瓣宽度
PetalWidth = sorted(data[:,3])
PetalWidth_list , PetalWidth_percentile = percentile(PetalWidth)

#画图：百分位数折线图
#萼片长度折线图
plt.scatter(SepalLength_list,SepalLength_percentile,label = '萼片长度')
plt.title('鸢尾花萼片长度百分位数')
plt.legend()
plt.show()
#萼片宽度折线图
plt.scatter(SepalWidth_list,SepalWidth_percentile,label = '萼片宽度')
plt.title('鸢尾花萼片宽度百分位数')
plt.legend()
plt.show()
#花瓣长度折线图
plt.scatter(PetalLength_list,PetalLength_percentile,label ='花瓣长度')
plt.title('鸢尾花花瓣长度百分位数')
plt.legend()
plt.show()
#花瓣宽度
plt.scatter(PetalWidth_list,PetalWidth_percentile,label = '花瓣宽度')
plt.title('鸢尾花花瓣宽度百分位数')
plt.legend()
plt.show()

'''
#位置度量：均值和中位数
#数据集鸢尾花萼片长度
'''
#print(SepalLength)
#均值
def mean(list):
    return sum(list)/len(list)
#中位数
def median(list):
    if len(list)%2 ==0:
        r = int(len(list)/2)
        return (list[r]+list[r+1])/2
    else:
        return list[r+1]
SepalLength_mean = mean(SepalLength)
SepalLength_median = median(SepalLength)
plt.plot(SepalLength,label = '鸢尾花萼片长度')
plt.scatter(len(SepalLength)/2,SepalLength_mean,marker='x',color = 'red',label = '鸢尾花长度均值')
plt.scatter(len(SepalLength)/2,SepalLength_median,marker='o',color = 'black',label = '鸢尾花长度中位数')
plt.legend()
plt.title('鸢尾花均值和中位数')
plt.show()

'''
#散布度量：极差和方差
'''

#极差
def rang_e(list):
    return max(list)-min(list)
#方差
def variance(list):
    list_mean = mean(list)
    varian = 0
    for i in range(len(list)):
        varian +=((list[i]-list_mean)**2)/(len(list)-1)
    return varian
#绝对平均偏差
def AAD(list):
    list_mean = mean(list)
    aad = 0
    for i in range(len(list)):
        aad += abs(list[i] - list_mean)/len(list)
    return aad
#中位数绝对偏差
def MAD(list):
    list_mean = mean(list)
    lis = []
    for i in range(len(list)):
        lis.append(abs(list[i]-list_mean))
    return median(lis)
#四分位极差
def IQR(list):
    list_list , list_percentile = percentile(list)
    i_75 = 0
    i_25 = 0
    flag1 = 0
    flag2 = 0
    for i in range(len(list_percentile)):
        if flag1*flag2 ==1  :
            break
        if str(list_percentile[i])=='75%':
            i_75 = i
            flag1=1
        if str(list_percentile[i])=='25%':
            i_25 = i
            flag2=1
    return list_list[i_75]-list_list[i_25]

#萼片长度极差
SL_range = rang_e(SepalLength)
#萼片长度标准差
SL_std  = math.sqrt(variance(SepalLength))
#萼片长度绝对平均差
SL_AAD =  AAD(SepalLength)
#萼片长度中位数绝对差
SL_MAD = MAD(SepalLength)
#萼片长度四分位数极差
SL_IQR = IQR(SepalLength)
print('度量   ，   萼片长度')
print('极差：',  SL_range )
print('std:',SL_std)
print('AAD:',SL_AAD)
print('MAD:',SL_MAD)
print('IQR:',SL_IQR)























































'''#导入鸢尾花数据集
iris = load_iris()
#频率和众数
target = iris['target']
#print(target)
dict = {'Setosa':0,'Versicolour':0,'Virginica':0}
for i in range(len(target)):
    if target[i]==0:
        dict['Setosa']+=1
    elif target[i]==1:
        dict['Versicolour']+=1
    else:
        dict['Virginica']+=1
'''























