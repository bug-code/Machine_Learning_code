import numpy
import matplotlib.pyplot as plt
import csv
import pandas
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#转置函数
def transpose(array):
    arr = []
    for j in range(len(array[0])):
        arr_j=[]
        for i in range(len(array)):
            arr_j.append(array[i][j])
        arr.append(arr_j)
    return arr
#定义画图函数
def draw_Eks(array):
    plt.plot(range(1,len(array)+1),array,label = '累计均方误差')
    plt.xlabel("训练次数")
    plt.ylabel("数据集单次训练累计均方误差")
    plt.legend()
    plt.show()

#BP神经网络学习算法
class BP_net(object):
    def __init__(self , eta , n_inter , q  ):
        self.eta  = eta
        self.inter = n_inter
        #self.d = d
        self.q = q
        #self.l = l

    def sum_arr(self,array):
        arr=[]
        for j in range(len(array[0])):
            aj=0
            for i in range(len(array)):
                aj+=array[i][j]
            arr.append(aj)
        return arr

    def Sigmoid(self,x): 
       return 1.0/(1+numpy.exp(-x))  

     #隐藏层输入
    def input_hide(self,x):
        return numpy.dot(x , self.v.T)
    #隐藏层输出
    def output_hide(self,x):
        Alpha = self.input_hide(x)
        Al_Gam=Alpha - self.Gamma
        self.bh =  self.Sigmoid(Al_Gam)
        return self.bh
    #输出层输入
    def input_out(self,x):
        bh = self.output_hide(x)
        return numpy.dot(bh,self.W.T)
    #输出层输出
    def predict(self, x):
        beta = self.input_out(x)
        bet_Thet = beta - self.Theta 
        return self.Sigmoid(bet_Thet)
    def fit(self , x , y):
        '''初始化输入层到隐藏层的权值
            假设单个训练样本xi有3个属性，隐藏层神经元有4个
                Vih = [
                        [v1 , v2 , v3 ]
                        [v4 , v5 , v6 ]
                        [v7 , v8 , v9 ]
                        [v10 , v11 , v12]
                       ]
                X = [
                        [x1 , x2 , x3]
                        ...........
                        [Xn-2 , Xn-1 , Xn ]
                    ]
            隐藏层输入
            Alpha = [X[0]*Vih[0] , X[0]*Vih[1] , X[0]*Vih[2] , X[0]*Vih[3]]
            隐藏层输出
            bh = Sigmoid(Alpha - Gamma)
            隐藏层到输出层权值,设隐藏层神经元为4个
            W = [
                 [w1 , w2 , w3 , w4]
                 [w5 , w6 , w7 , w8]
                 [w9 , w10 , w11 , w12]
                ]
            输出层输入
            beta = [W[0]*bh , w[1]*bh , w[2]*bh ]
            输出层输出
            y = Sigmoid[beta - Theta]
        '''
        self.v = numpy.random.rand(self.q , x.shape[1])                                                                                                                                                                                                                         
        #初始化隐藏层神经元阈值
        self.Gamma = numpy.random.rand(self.q) 
        #初始化隐藏层到输出层权值
        self.W = numpy.random.rand(y.shape[1],self.q)
        #初始化输出层神经元阈值
        self.Theta = numpy.random.rand(y.shape[1])
        #创建总均方误差统计数组
        self.Eks = []
        #执行self.inter次训练
        for N in range(self.inter):
            #单次训练错误次数初始化为0
            Ek= []
            #从训练集中抽取单个训练样本
            for xi , target in zip(x,y):
                #获得输出层输出
                self.y=self.predict(xi)
                #计算单次输出均方误差
                for i in range(len(target)):
                    Ek_ = self.y[i]-target[i]
                    Ek.append(Ek_)
                #调整隐藏层到输出层权值和阈值
                Sigma_Whj_gj=[] #二维数组
                for n in range(len(target)):
                    #计算gj
                    gj = self.y[n]*(1-self.y[n])*(target[n]-self.y[n])
                    #计算某个神经元下的Whj_gj
                    Sigma_Whj_gj_n = self.W[n]*gj
                    Sigma_Whj_gj.append(Sigma_Whj_gj_n)
                    #计算Δθ
                    Delta_Theta = (-1)*self.eta*gj
                    #更新θ
                    self.Theta[n] +=Delta_Theta
                    #计算Δw
                    Delta_w = [self.eta*gj*self.bh[i] for i in range(len(self.bh))]
                    #更新W
                    self.W[n] += Delta_w

                #调整输入层到隐藏层的权值和阈值
                #计算各个ΣWhj_gj的值
                Sigma_W_g = self.sum_arr(Sigma_Whj_gj)
                #计算隐藏层各个神经元的eh
                eh=[]
                for n in range(len(Sigma_W_g)):
                    eh_ = self.bh[n]*(1-self.bh[n])*Sigma_W_g[n]
                    eh.append(eh_)
                #计算Δγ
                Delta_Gamma =[ (-1)*self.eta*eh[n] for n in range(len(eh))] 
                #更新阈值γ
                self.Gamma +=Delta_Gamma
                #计算Δv
                for n in range(len(self.v)):
                    Delta_v = self.eta*eh[n]*self.v[n]
                    self.v[n] +=Delta_v
            self.Eks.append(numpy.average(0.5*sum(numpy.square(Ek))))
#打开训练样本，取出数据

read_file = pandas.read_csv("F://code//my_project//BP_net//BP_net//train.csv",header=0)
target= read_file.iloc[0:,[3,4]].values
x = read_file.iloc[0:,[0,1,2]].values

#数据归一化处理
x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(-1,1))
x=x_scaler.fit_transform(x)
target = y_scaler.fit_transform(target)

x_train=x[0:15]
target_train=target[0:15]
x_test=x[15:]
target_test=target[15:]


#设置学习率，训练次数，隐藏层神经元个数
bp_net =BP_net(eta = 0.02 , n_inter = 2000 , q =6)
#训练样本
bp_net.fit(x_train ,target_train )
draw_Eks(bp_net.Eks)

#仿真输出和实际输出对比图
netout_target = []
for i in range(len(x_test)):    
    netout_target.append(bp_net.predict(x_test[i]))

netout_target = y_scaler.inverse_transform(netout_target)
netout_target = transpose(netout_target)
target_test = y_scaler.inverse_transform(target_test)
target_test = transpose(target_test)
'''print(netout_target[0])
print(target_test[0])
print(netout_target[1])
print(target_test[1])
'''
#客运量预测图
plt.plot(netout_target[0], marker = 'o', label = '预测值')
plt.plot(target_test[0], marker = 'x', label = '实际值')
plt.xlabel('时间')
plt.ylabel('客运量')
plt.title('客运量对比图')
plt.legend()
plt.show()
#货运量预测图
plt.plot(netout_target[1], marker = 'o', label = '预测值')
plt.plot(target_test[1], marker = 'x', label = '实际值')
plt.xlabel('时间')
plt.ylabel('客运量')
plt.title('货运量对比图')
plt.legend()
plt.show()











































                

                

                  




                
                
