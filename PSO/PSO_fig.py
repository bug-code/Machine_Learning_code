from mpl_toolkits.mplot3d  import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
#设置中文编码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# #画出函数三维图像
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-2, 2, 0.05)
# Y = np.arange(-2, 2, 0.05)
# # 在X,Y范围内建立网格点
# X, Y = np.meshgrid(X, Y)
# #图像函数
# Z = np.sin(np.sqrt(X**2+Y**2))/np.sqrt(X**2+Y**2)+np.exp((np.cos(2*np.pi*X)+np.cos(2*np.pi*Y))/2)-2.71289
# #rstride:指定行跨度 cstride:指定列跨度
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
# plt.show()

# a = np.random.uniform(-2,2,(20,2))
# x = a[0][0]
# y = a[0][1]
# print(a,x,y)


class basicPSO(object):
    
    '''设置学习因子，惯性权重，迭代次数，种群大小'''
    def __init__(self, c1 , c2 , w , iteration,pops):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.iteration = iteration 
        self.pops = pops
    #随机创建种群数量为的20粒子群，其中包括每个粒子位置，速度
        self.x =  np.random.uniform(-2,2,(self.pops,2))
        self.v =  np.random.uniform(-4,4,(self.pops,2))
        # self.Lbest_fitness = np.zeros(self.pops)
        self.HPLbest =self.x 
        self.is_change = 1    
        # self.HPGbest = self.x[0]
        
    ''' 初始化评估每个粒子适应度，并获得全场最佳'''
    def get_fitness(self,Position):
        Lbest_fitness = []
        for i in range(self.pops):
            X = Position[i][0]
            Y = Position[i][1]
            Z = np.sin(np.sqrt(X**2+Y**2))/np.sqrt(X**2+Y**2)+np.exp((np.cos(2*np.pi*X)+np.cos(2*np.pi*Y))/2)-2.71289
            Lbest_fitness.append(Z)
        i_best = Lbest_fitness.index(max(Lbest_fitness))
        PGbest = Position[i_best]
        return Lbest_fitness , PGbest

    '''更新位置，速度 '''
    def update_x_v(self):
        ''' 更新速度,位置'''
        for i in range(self.pops):
            random1 = random.random()
            random2 = random.random()
            self.v[i]= self.w*self.v[i] +self.c1*random1*(self.HPLbest[i] - self.x[i]) +   self.c2*random2*(self.HPGbest - self.x[i])
            if self.v.all() > 4 :
                pass
            # self.v[i] = self.w*self.v[i][1] +self.c1*random1*(self.HPLbest[i][1] - self.x[i][1]) +   self.c2*random2*(self.HPGbest[i][1] - self.x[i][1])
            self.x[i] = self.x[i] +self.v[i]
        
    # ''' 获得全场最佳最终结果'''
    def get_PGbest(self):
        '''初始化获得全体粒子适应度值，和最佳适应度值的位置'''
        self.Lbest_fitness , self.HPGbest = self.get_fitness(self.x)
        iter = 0
        while iter<self.iteration :
            self.update_x_v()
            #获得更新后的X的全体粒子适应度，和更新后粒子最佳适应度位置
            self.Lbest_fitness , self.PGbest = self.get_fitness(self.x)
            #获得历史最佳全体粒子适应度值，和历史最佳的粒子中最佳适应度位置
            self.HLbest_fitness , self.HPGbest = self.get_fitness(self.HPLbest) 
            for i in range(self.pops):
                #如果更新后的xi处适应度值比该处历史最佳大，则更新该处历史最佳的位置
                if self.Lbest_fitness[i] > self.HLbest_fitness[i] :
                    self.HPLbest[i]  = self.x[i]
            #更新历史中全部粒子的最佳位置
            self.HLbest_fitness , self.HPGbest0 = self.get_fitness(self.HPLbest)
            if (self.HPGbest0 == self.HPLbest).all() :
                self.is_change = 0
            else:
                self.HPGbest = self.HPGbest0

        PGbest_fitness = self.get_fitness(self.HPGbest)
        return self.HPGbest , PGbest_fitness

pso = basicPSO(2,2,0.5,1,1)

position , position_fitness = pso.get_PGbest()
print(position , position_fitness)

            
                
        

    
                
            
    
        
        

