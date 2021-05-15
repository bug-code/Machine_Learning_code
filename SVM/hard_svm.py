'''
SOM算法流程：
    1，构造函数设置循环优化次数。
    2.在（0~C）范围内初始化拉格朗日乘子a，位移b，
    外循环：交替遍历 整个样本集 和 遍历非边界样本集 选取违反KKT条件的a_i作为第一个变量
        KTT条件：a_i = 0 ➡ y^i( W^T*x^i + b )>=1
                a_i = C ➡y^i( W^T*x^i + b ) <= 1
                0<a_i<C ➡y^i( W^T*x^i + b ) =1
        内循环：选取与a_i有较大变化的a_2
            因为：a2_new = a2_old +y2(E1-E2)/η    //计算新a2的公式
                  Ei = f(xi)-yi /实际值-预测值
                  η = x1^T*x1+x2^T*x^2-2x1^T*x2 /常数
                  所以a2依赖于|E1-E2|，选取a2使得|E1 - E2|最大
                   1，当E1为正时选取a2使得E2最小
                   2，当E1为负时选取a2使得E2最大
            ///一对拉格朗日乘子选取完成////
            固定其他参数单独对a1和a2进行优化，当拉格朗日乘子α中每一个都优化完成就说明整个拉格朗日乘子优化完成
            对于a2_new = a2_old +y2(E1-E2)/η给定的约束条件
                0<=a_i<=C
                Σ（i:1➡m）a_i*y_i = 0
                因为固定其他参数优化a1和a2，所以上述约束条件变为：
                0<=a_i<=C
                a1*y1+a2*y2 = -Σ（i:3→m）a_i*y_i
                当y1!=y2时：L = max(0,Σ（i:3→m）a_i*y_i )  H = min( C , C+Σ（i:3→m）a_i*y_i )
                当y1=y2时： L = max(0,-Σ（i:3→m）a_i*y_i  - C ) H = min(C,-Σ（i:3→m）a_i*y_i)
                添加约束条件之后，对于未修剪得a2进行调整
                当a2_new>H时 ，a2_new = H
                当L<=a2_new<=H时，a2_new = a2_new
                当a2_new<L时 ， a2_new = L
                更新a1_new:
                    因为a1_old*y1+a2_old*y2 = a1_new*y1+a2_new*y2
                    推出:
                        a1_new = a1_old+y1*y2*(a2_old-a2_new)
            ///选取得一对a1和a2优化完成///
            更新b
            当0<a1_new<C时：b = - E1-y1*x1^T*x1(a1_new-a1_old)-y2*x2^T*x1(a2_new-a2_old)+b_old = b1
            当0<a2_new<C时：b = - E2-y1*x1^T*x2(a1_new-a1_old)-y2*x2^T*x2(a2_new-a2_old)+b_old +b2
            当0<ai_new<C，i= 1,2时：b=b1=b2
            当ai_new同时不满足0<ai_new<C，i= 1,2时：b = (b1+b2)/2
    循环以上过程，当达到循环次数或者拉格朗日乘子无太大变化时结束循环
    根据优化后得拉格朗日乘子计算出要求的W*
    W* = Σa_i*y_i**x_i    i:0➡m       最终的b*从求出的支持向量中求其平均值
    //////////////以上SVM算法全部完成///////////////////
资料：https://www.bilibili.com/video/av28186618
      https://blog.csdn.net/luoshixian099/article/details/51227754
      https://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html

'''
import numpy
import random
import pandas
import matplotlib.pyplot as plt
#设置中文编码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class SVM(object):
    def __init__(self,iteration):
        self.iteration = iteration
    #拉格朗日乘子a未全部更新完成的预测模型，判断样本类别
    def fxi(self,alphas,labels,datas,data_i,b):
        ai_yi = numpy.multiply(alphas,labels) #拉格朗日乘子中每个α与数据集的对应标签相乘
        w = numpy.dot(ai_yi,datas) 
    
        return numpy.dot(w,data_i)+b
    #交替从 整个样本集 和 遍历非边界样本集 选取违反KKT条件的a_i作为第一个变量
    #flag=1表示从样本集中选取，flag=0表示从非边界样本集中选取
    def select_a1(self,flag,alphas,labels,datas,b):
        #从样本集中选取
        if flag==1:
            locate = -1    #标志位，当返回值为-1时已经全部满足kkt条件
            for i in range(len(alphas)):
                kkt = labels[i]*self.fxi(alphas,labels,datas,datas[i],b)
                if alphas[i]==0 and kkt<1 or kkt>1 or alphas[i]>0 and kkt!=1:
                    locate = i
                    flag = 0
                    break
        #从非边界样本集中选取
        else:
            locate = -1
            for i in range(len(alphas)):
                kkt = labels[i]*self.fxi(alphas,labels,datas,datas[i],b)
                if alphas[i]>0 and kkt!=1:
                    locate = i
                    flag = 1
                    break
        return locate ,flag
    #选取与a1差距最大的a2
    def select_a2(self,alphas,labels,datas,b,locate_1):
        E1 = self.fxi(alphas,labels,datas,datas[locate_1],b) - labels[locate_1]
        distance = 0
        for i in range(len(alphas)):
            if i != locate_1:
                E2 = self.fxi(alphas,labels,datas,datas[i],b) - labels[i]
                new_distance =  abs(E1-E2)
                if new_distance>distance:
                    distance = new_distance
                    locate_2 = i
        return locate_2
    #计算除固定的a1和a2的Σai*yi
    def Sigma_ai_yi(self , alphas, labels , locate_1 , locate_2):
        Sigma_ai_yi = 0        
        for i in range(len(alphas)):
            if i != locate_1 and i != locate_2:
                Sigma_ai_yi += alphas[i]*labels[i]
        return Sigma_ai_yi
    #选取完a1,a2后优化a1,a2
    def update_a1_a2(self,alphas,labels,locate_1,locate_2,datas,b):
        a1_old = alphas[locate_1]
        a2_old = alphas[locate_2]
        # if labels[locate_1] != labels[locate_2] :
        #     L = max( 0 , a2_old-a1_old )  
        #     H = min( 0.6, 1+a2_old-a1_old )
        # else:
        #     L = max(0 ,a2_old+a1_old -0.6 ) 
        #     H = min(0.6 , a2_old+a1_old  )
        #计算新a2
        E1 = self.fxi(alphas,labels,datas,datas[locate_1],b) - labels[locate_1]
        E2 = self.fxi(alphas,labels,datas,datas[locate_2],b) - labels[locate_2]
        eta = numpy.dot(datas[locate_1],datas[locate_1].T)+numpy.dot(datas[locate_2],datas[locate_2].T) - 2*numpy.dot(datas[locate_1],datas[locate_2].T)
        a2_new = a2_old +labels[locate_2]*(E1-E2)/eta
        '''根据a2_new对a2_new进行修剪
                当a2_new>H时 ，a2_new = H
                当L<=a2_new<=H时，a2_new = a2_new
                当a2_new<L时 ， a2_new = L
        '''
        # if a2_new > H:
        #     a2_new = H
        # if a2_new<L:
        #     a2_new = L
        #//////////针对a2修剪完成
        #更新a1
        a1_new = a1_old+labels[locate_1]*labels[locate_2]*(a2_old-a2_new)
        '''更新b
            当0<a1_new<C时：b = - E1-y1*x1^T*x1(a1_new-a1_old)-y2*x2^T*x1(a2_new-a2_old)+b_old = b1
            当0<a2_new<C时：b = - E2-y1*x1^T*x2(a1_new-a1_old)-y2*x2^T*x2(a2_new-a2_old)+b_old = b2
            当0<ai_new<C，i= 1,2时：b=b1=b2
            当ai_new同时不满足0<ai_new<C，i= 1,2时：b = (b1+b2)/2
        '''
        b1 = -E1-labels[locate_1]*numpy.dot(datas[locate_1],datas[locate_1])*(a1_new-a1_old)-labels[locate_2]*numpy.dot(datas[locate_2],datas[locate_1])*(a2_new-a2_old) +b
        b2 = -E2-labels[locate_1]*numpy.dot(datas[locate_1],datas[locate_2])*(a1_new-a1_old)-labels[locate_2]*numpy.dot(datas[locate_2],datas[locate_2])*(a2_new-a2_old)+b
        # if a1_new > 0 :
        #     b = b1
        # elif a2_new > 0:
        #     b =  b2
        # else:
        b = (b+b1+b2)/3
        
        alphas[locate_1] = a1_new
        alphas[locate_2] = a2_new
        return b
    def SMO(self,labels,datas):
        #初始化拉格朗日乘子α和b
        m  = len(labels)
        self.alphas = numpy.random.rand(1,m)[0]
        self.b= 0
        for itera in range(self.iteration):
            '''选择a1,选择a2
            '''
            flag_exit = 0
            flag_a1 = 1
            for i in range(m):
                locate_1 , flag_a1   = self.select_a1(flag_a1,self.alphas,labels,datas,self.b)
                #当找不到可以进行优化的α时，表示优化完成，终止所有循环
                if locate_1 == -1:
                    flag_exit =1
                    break
                #内循环选择a2
                locate_2 = self.select_a2(self.alphas,labels,datas,self.b,locate_1)
                #对a1,a2,b进行优化更新
                self.b = self.update_a1_a2(self.alphas,labels,locate_1,locate_2,datas,self.b)
            if flag_exit==1:
                break
        #以上优化更新迭代完成，之后计算W*
        self.w = numpy.dot(numpy.multiply(self.alphas,labels),datas)

read_file = pandas.read_csv('testSet.csv')
labels= read_file.iloc[0:100,2].values
datas = read_file.iloc[0:100,[0,1]].values


# b是常量值， alphas是拉格朗日乘子

svm = SVM(200)
svm.SMO(labels,datas)
#计算超平面直线
y1 = -4
x1 = (-svm.b-svm.w[1]*y1)/svm.w[0]
y2 = 4
x2 = (-svm.b-svm.w[1]*y2)/svm.w[0]

# 画出样本数据集图像
for i in range(len(labels)):
    if labels[i] > 0 :
        plt.scatter(datas[i,0],datas[i,1],color = 'black',marker='o')
    else:
        plt.scatter(datas[i,0],datas[i,1],color = 'red',marker='x')
plt.plot([x1,x2],[y1,y2])
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.title('训练样本图像')
plt.show()



            




            
            
            


        

        
    