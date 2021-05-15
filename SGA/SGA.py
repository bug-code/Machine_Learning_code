'''
遗传算法流程：
    （1）随机产生初始解群体，并对其进行编码
                （1）十进制编码
                        优点：无需解码
                        缺点：突变可能性太多，有9种。交换太粗略，在多代循环后收敛太慢
                （2）二进制编码
                        优点：突变取反即可，唯一确定。交换时可确定精度变化，在多代循环
                        后收敛速度较快
                        缺点：需要解码，增加了运算量
                （3）间接二进制码（本算法所采用的方法）
                            d为解的精确度，[U_min , U_max]为解的定义域范围，码位公式
                                2^m<(U_max-U_min)/d<=2^(m+1)
                            则需要编码的位数为m+1
    循环：
    （2）设定适应性函数
            当使用轮盘赌的选择复制算子时：以适应性函数值大小来确定概率，决定哪些个体以
            多少概率存活下去，以多少概率消亡
    （3）复制。对经过适应性函数筛选过的个体有策略的进行复制，使得种群数量保持不变
        （选择复制算子，本算法采用锦标赛选择法）
    （4）随机选择一定数量个体，并随机选择个体染色体其中几个基因进行交换（繁殖）
        （交换算子）
    （5）随机选择一定数量个体，对其中的染色体随机选择几个基因进行补变换（变异）
        （变异算子）
    （6）解码
            间接二进制码解码公式：
                        x = U_min + (U_max - U_min)/(2^m - 1)Σ（b_i*2^(i-1)）
                        i∈[1,m]

    （7）判断新产生的种群是否满足需要，来决定是否终止循环
    参考资料：https://www.bilibili.com/video/av41127323
            https://blog.csdn.net/u010451580/article/details/51178225
            https://www.cnblogs.com/legend1130/archive/2016/03/29/5333087.html
            https://wenku.baidu.com/view/031acc42657d27284b73f242336c1eb91a3733ad.html
            https://www.jianshu.com/p/4bc8b4d0ae61
算法要解决的问题：
        f(x,y) = 21.5+xsin(4Πx)+ysin(20Πy)
        该函数从值域从正无穷到负无穷，定义域从正无穷到负无穷
        求：
            在给定区间内，满足给定精度内的最大值（最优化问题）
'''
import random
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class SGA(object):
    '''初始化种群X与y的区间和精度'''
    def __init__(self , start_x , end_x  , start_y , end_y,d_x , d_y ,n,iteration):
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.d_x = d_x
        self.d_y = d_y
        self.n = n      #种群数
        self.iteration = iteration      #迭代数
    '''计算符合要求的二进制码位数'''
    def binary_bites(self,start , end , d ):
        m = 0
        while 2**m < (end-start)/d:
            m +=1
        return m
    '''初始化种群，随机产生n个群体，基因型随机self.genes'''
    def gene_codes(self):
        #对种群x进行间接二进制编码时所需要的位数
        self.m_x = self.binary_bites(self.start_x,self.end_x,self.d_x)
        #对种群y进行间接二进制编码时所需要的位数
        self.m_y = self.binary_bites(self.start_y,self.end_y,self.d_y)
        #随机产生n组【x,y】基因型
        genes  = [[[random.randint(0, 1) for i in range(self.m_x)],[random.randint(0, 1) for i in range(self.m_y)]] for j in range(self.n)]
        return genes 

    '''对基因解码,根据基因型计算出每个个体的表现型'''
    def gene_decode(self ,genes):
        phenotype = []
        for i in range(len(genes)):
            li = []
            genes_x = genes[i][0]
            genes_y = genes[i][1]
            Sigma_x = 0
            Sigma_y = 0
            for j in range(len(genes_x)):
                Sigma_x +=genes_x[j]*(2**(len(genes_x)-j-1))
            x = self.start_x+(self.end_x - self.start_x)/(2**self.m_x-1)*Sigma_x
            li.append(x)
            for j in range(len(genes_y)):
                Sigma_y +=genes_y[j]*(2**(len(genes_y)-j-1))
            y = self.start_y+(self.end_y - self.start_y)/(2**self.m_y-1)*Sigma_y
            li.append(y)
            phenotype.append(li)
        return phenotype
    
    '''适应性算子:
            （1）根据表现型，求出每个个体得适应度
            （2）对适应度进行重排序，再根据适应度得重排序对个体进行重排序
            （3）针对重排序后得种群个体进行选择复制'''

    def adaption(self,phenotype,genes):
        adapt = []
        #计算每个个体的适应性
        for i in range(len(phenotype)):
            adaptive = 21.5+phenotype[i][0]*math.sin(4*math.pi*phenotype[i][0])+phenotype[i][1]*math.sin(20*math.pi*phenotype[i][1])
            adapt.append(adaptive)
        #依据个体适应性对个体（基因型）,与对应的适应性进行排序进行排序
        for i in range(len(adapt)):
            for j in range(i,len(adapt)):
                if adapt[i]<=adapt[j]:
                    genes[i] , genes[j] = genes[j] , genes[i]
                    adapt[i] , adapt[j] = adapt[j] , adapt[i]
        #返回排序后的适应度
        return  adapt

    '''选择复制算子（锦标赛选择法）'''
    def chose_copy(self , genes , adapt):
        #初始化下一代基因型
        CC_genes = []
        #设置选择次数
        for i in range(len(genes)):
            #从上一代基因型中选择int(len(genes)/4)个基因型参加竞标赛
            champ_member = random.sample(adapt,k = int(len(adapt)/4))
            winner = max(champ_member)
            locate = adapt.index(winner)
            CC_genes.append(genes[locate])
        return CC_genes

    '''
    交换算子，随机选择种群中的两个基因型，随机对基因型上对应的m个基因进行交换
    注意：每次须选择不同的个体进行交换，每次进行交换的位置须对应。参加过交换的
          基因型不应当参加到下一次基因交换中 
    '''
    def gene_change(self,gene):
        #-----------------------基因交换-----------------------------------------#
        #交换概率设置75%
        is_exchange = random.randint(0,4)
        if is_exchange:
            #设置交换数量，种群数量的一半
            exchange_N = int(len(gene)/2)
            if exchange_N%2 != 0:
                exchange_N +=1
            #在父代中随机选择一半的种群进行基因交换
            exchange_members = random.sample(gene , k = exchange_N)
            #在父代中移除用于交换的基因型
            for i in range(len(exchange_members)):
                gene.remove(exchange_members[i])
            #基因交换
            while exchange_members:
                exchange_member = random.sample(exchange_members,k=2)
                for i in range(len(exchange_member)):
                    exchange_members.remove(exchange_member[i])
                #基因编码位数的三分之一进行交换
                exchange_start_x = random.randint(int(self.m_x/2),int(self.m_x*2/3))
                exchange_start_y = random.randint(int(self.m_y/2),int(self.m_y*2/3))
                #种群X上基因位交换
                exchange_part_x = exchange_member[0][0][exchange_start_x:exchange_start_x+int(self.m_x/3)]
                exchange_member[0][0][exchange_start_x:exchange_start_x+int(self.m_x/3)] = exchange_member[1][0][exchange_start_x:exchange_start_x+int(self.m_x/3)]
                exchange_member[1][0][exchange_start_x:exchange_start_x+int(self.m_x/3)] = exchange_part_x
                #种群y上基因位交换
                exchange_part_y = exchange_member[0][1][exchange_start_y:exchange_start_y+int(self.m_y/3)]
                exchange_member[0][1][exchange_start_y:exchange_start_y+int(self.m_y/3)] = exchange_member[1][1][exchange_start_y:exchange_start_y+int(self.m_y/3)]
                exchange_member[1][1][exchange_start_y:exchange_start_y+int(self.m_y/3)] = exchange_part_y
                #种群x , y交换完成，添加回父代中
                gene.append(exchange_member[0])
                gene.append(exchange_member[1])
            #-----------------基因交换完成------------------#

        #----------------------基因变异-----------------------------#
        '''
        基因变异:
                种群完成选择复制，基因交换后，概率随机选择相应种群个体，对其基因随机选择位置变异
                变异算子：随机选取基因序列的两个位置k和m，逆转其k~m间的城市编号
        '''
        #设置基因变异概率10%
        is_variation = random.randint(0,10)
        if 0==is_variation:
            #随机选取种群数量的10%进行变异
            variat_members = random.sample(gene,k = int(len(gene)*0.1))
            #原始种群中删除要进行变异的个体
            for i in range(len(variat_members)):
                gene.remove(variat_members[i])
            #对需要变异的个体进行基因变异
            for variat_member in variat_members:
                '''随机产生1-len(variat_members)之间的两个整数m ， k
                    如果m<k,则将个体中对应的基因进行reverse变换
                '''
                #对种群x进行基因变异
                m_x = random.randint(int(len(variat_member)/2),len(variat_member))
                k_x = random.randint(int(len(variat_member)/2),len(variat_member))
                if m_x<k_x:
                    gene_reverse = variat_member[0][m_x:k_x]
                    gene_reverse.reverse()
                    variat_member[0] = variat_member[0][0:m_x]+gene_reverse+variat_member[0][k_x:]
                #对种群y进行基因变异
                m_y = random.randint(int(len(variat_member)/2),len(variat_member))
                k_y = random.randint(int(len(variat_member)/2),len(variat_member))
                if m_y<k_y:
                    gene_reverse = variat_member[1][m_y:k_y]
                    gene_reverse.reverse()
                    variat_member[1] = variat_member[1][0:m_y]+gene_reverse+variat_member[1][k_y:]
                #个体变异完成，添加回基因种群中
                gene.append(variat_member)
        return gene
    

    '''
    种群迭代
    '''
    def gene_iteration(self):
        max_adapt = []
        #初始化第一代
        genes = self.gene_codes()
        #开始迭代
        for i in range(self.iteration):
           #计算基因表现型
           phenotype = self.gene_decode(genes)
           #计算种群的适应性
           adapt = self.adaption(phenotype,genes)
           #记录本代种群适应性的最大值
           max_adapt.append(max(adapt))
           
           #种群进化
           genes = self.chose_copy(genes,adapt)  #选择复制父代基因种群
           #基因交换和基因变异
           genes = self.gene_change(genes)
        
        #迭代完成，返回每一代种群中适应度最大的列表记录
        return max_adapt


'''test:
        x∈[-5.23,2.45],y∈[3.56,6.21] 
        问题函数：f(x,y) = 21.5+xsin(4Πx)+ysin(20Πy)
'''
#以下参数均可随意设置
start_x = -5.23
end_x = 2.45
start_y =3.56
end_y = 6.21
d_x = 0.01  #x精度
d_y = 0.01  #y精度
n = 50     #每代种群数量
iteration =30   #迭代次数

sga = SGA(start_x , end_x  , start_y , end_y,d_x , d_y ,n,iteration)
max_adapt = sga.gene_iteration()
plt.plot(max_adapt,marker ='x',label = '适应度最大值变化')
plt.xlabel("迭代次数")
plt.ylabel("适应度最大值")
plt.legend()
plt.show()






        



                    





                
            

