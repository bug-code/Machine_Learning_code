#created by yangpeng 
'''
数据集分三个文件：ratings.dat、movies.dat、users.dat
ratings.dat

            用户ID | 电影ID | 评分 | 时间戳

movies.dat
            电影ID | 电影名称 | 电影类别

users.dat
            用户ID | 性别 | 年龄 | 职业 | 邮编

1.首先读入三个数据集的数据
2.将三个数据集合成一个数据集
3.在合成一个数据集时，将无用属性剔除减少计算量
4.需要剔除的属性有：用户评分时间（都在2000年）、标题（训练、测试时用不上）、邮编
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy    as np
import os
import  math
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from  collections import OrderedDict
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from scipy.sparse import csr_matrix
from collections import Counter

#设置中文编码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#数据预处理
#读取文件数据
'''
转换为CSV文件
# users_path = os.getcwd()+'/users.dat'
# users_data = pd.read_table(users_path , header=None  )                                                                                                   ///转换为CSV文件
# users_data = users_data.to_csv('users.CSV' , index=False, header = False , sep=',')
'''
def table_join( users_data, movies_data , ratings_data): 
    #用户表中性别转换为数字
    gender_map = {'F':0 , 'M':1}
    users_data['gender'] = users_data['gender'].map(gender_map)
    #ratings_dat、users_dat与movies_dat 合成一张表
    RM_table = pd.merge(ratings_data,movies_data,on=['movieID'],how="left")
    RM_table = RM_table.drop(['timestamp' , 'title'],axis=1)
    RMU_table = pd.merge(RM_table , users_data ,on=['userID'] , how="left" )
    RMU_table = RMU_table.drop(['zip-code'],axis=1)
    #先保存合一表
    np.savetxt("RMU.CSV",RMU_table ,delimiter=',' , fmt='%s') 
    return RMU_table 
'''将电影类别转换为6位的十进制数'''
def genres2digital(genres_dat):
    genres_li = ['Action' , 'Adventure' , 'Animation' , 'Children\'s' ,'Comedy' , 'Crime'  , 'Documentary' , 'Drama' , 'Fantasy' , 'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' , 'Thriller' , 'war' , 'Western' ]
    #待返回的十进制电影类别编码
    gen_dat_update = [ ]    
    for genre in genres_dat:
        tmp_genredat = genre.split('|')
        #某部电影的十进制编码
        genres_ten = 0
        for genre_tmp in tmp_genredat:
            for genres_tmp in genres_li:
                if genre_tmp == genres_tmp:
                    genretmp_index = 17-genres_li.index(genres_tmp)
                    genres_ten +=  math.pow(2,genretmp_index)
        gen_dat_update.append(genres_ten)
    return gen_dat_update
# new_genres = genres2digital(RMU_data['genres'])   
# #将十进制genres替换原数据
# RMU_data['genres'] = new_genres

# #保存为new_RMU文件
# np.savetxt("new_RMU.CSV",RMU_data,delimiter=',' , fmt='%s') 
'''
构造用户评分矩阵
'''
#统计总用户数和总电影数,构造用户评分矩阵
def URmat_fuc(rmu_data ):
    sum_users =rmu_data.userID.unique().shape[0]
    sum_movies =int( rmu_data["movieID"].max())
    # print(sum_users , sum_movies)
    #创建用户评分矩阵
    UR_mat = np.zeros((sum_users,sum_movies))

    for row in RMU_data.itertuples() :
        UR_mat[int(row[1])-1, int(row[2])-1] = int(row[3])
    return UR_mat
#计算相似度
def calc_CosSimilar(ur_mat):
    #基于用户的协同过滤
    user_similar = cosine_similarity(ur_mat , dense_output=True)
    movies_similar = cosine_similarity(ur_mat.T , dense_output=True)
    return user_similar , movies_similar
'''
推荐系统：
    1.获得用户ID
    2.计算该用户与其他用户的相似度
    3.使用的算法Threshold-based neighborhoods选择出若干个用户作为参考进行推荐
    4.针对搜索推荐。输入电影名，推荐包含该类型的相似电影按评分推荐  (√)
    5.针对新用户使用user数据集，依据user相似矩阵进行推荐
'''


'''
模块一：搜索模块
'''
#输入电影名称推荐电影模块
#构造电影ID 、 电影名称 、 平均分的dataframe

def search_name(movies_data  , name , ru_mat):
    title_flag = input("search by title:")

    # #矩阵中用户总数  矩阵列 、行
    sum_users = np.size(ru_mat,1)
    sum_movies = np.size(ru_mat ,0)
    #保存平均评分
    aver_lis = []
    for i in range(sum_movies):
        movie_averpoint = sum(ru_mat[i])/sum_users
        aver_lis.append(movie_averpoint)
    #movies_data增加一列，最终输出 电影ID 、 电影标题 、 电影类别 、 电影平均评分
    movies_data['averpoint'] = aver_lis[:3883]
    #根据movies_data查找具有该类别的电影，再根据平均评分排序截取前十数据返回
    #先查找电影名称、再查找电影类型，均无则返回错误告知
    if title_flag == 'y'  or title_flag=='Y':
        reco_movies = movies_data[movies_data['title'].str.contains(pat=name)]
        
    else :
        reco_movies = movies_data[movies_data['genres'].str.contains(pat=name)]
        
    reco_movies=reco_movies.sort_values(by = 'averpoint' , ascending =False)[:10]
    return reco_movies

'''
模块二：用户推荐模块
    #新用户推荐模块、推荐10部电影
    #使用user数据集，构造用户个人信息矩阵，求余弦相似度
    #从相似度最高的用户中选择推荐10部电影
        #1.根据用户ID查找3个相似用户，对相似用户降序排序，保存相似用户ID
        #2.从用户评分表找到相似用户观影数据 ， 找到用户未看过的电影
        #3.对相似用户使用   用户之间相似度*0.7+0.3*电影评分 = 电影推荐得分
        #4.对用户未看过的电影推荐评分降序排序，保存电影ID，使用movie数据集显示前10条
        #user相似度表，对loginID相似度排序
        #相似矩阵转成dataframe
'''   
def new_userreco(users_data , loginID):
    gender_map = {'F':0 , 'M':1}
    users_data['gender'] = users_data['gender'].map(gender_map)
    users_data = users_data.drop(['zip-code'],axis=1)
    users_mat = users_data.iloc[:,:].values
    users_mat=preprocessing.scale(users_mat)
    #构造相似度矩阵
    newuser_similar = cosine_similarity(users_mat , dense_output=True)
    fig , ax = plt.subplots(1,1)
    ax.set_title('用户评分矩阵可视化' , fontsize = 12)
    ax.matshow(newuser_similar)
    plt.show()
    NU_silimar_df = pd.DataFrame(newuser_similar)

    #截取该用户与其他用户的相似度
    user_similar=NU_silimar_df.iloc[loginID-1]
    #user_similar中的列名称即为用户ID-1 ， 先获取用户ID-1
    user_similar_dict = user_similar.to_dict()
    #与logID用户相似的用户存放字典中，再根据字典中的value值排序取其前三
    user_similar_dict = sorted(user_similar_dict.items() , key=lambda x: x[1] , reverse=True )[1:4]
    #记录当下推荐的电影数
    sum_recmovies = 0
    #新建推荐电影ID空字典
    rec_moviesID  = []
    for x in range(3):
        #获取的是相似用户ID-1
        similar_userID = user_similar_dict[x][0]
        #当下第i个相似用户的相似度
        simila = user_similar_dict[x][1]
        #在用户评分矩阵中，该相似用户所在的行就是相似用户ID-1=similar_userID
        similuser_rating = UR_mat[similar_userID]
        #推荐10部电影，从当前第i个相似用户的用户评分表中找到看过的电影，并计算电影推荐得分，公式：用户相似度*0.7+电影评分*0.3
        for i in range(len(similuser_rating)):
            if sum_recmovies < 10:
                #矩阵是从0开始，真实movieID要加1
                if similuser_rating[i] != 0:
                    score = simila*0.7+0.3*similuser_rating[i]
                    rec_moviesID.append(([i+1] , score))
                    sum_recmovies +=1
                    #每次加入新推荐电影，就对字典排序，为推荐电影满10个时插入更高评分的电影准备
                    rec_moviesID =  sorted(rec_moviesID, key=lambda x: x[1] , reverse=True )

            else:
                if similuser_rating[i] != 0:
                    score = simila*0.7+0.3*similuser_rating[i]
                    if rec_moviesID[9][1] < score:
                        #删除字典中电影得分最少的元组，加入评分更高的元组，使推荐电影的评分在所有的电影中都是最高的
                        rec_moviesID[9]=([i+1] , score)
                        rec_moviesID = sorted(rec_moviesID , key=lambda x: x[1] , reverse=True )
    #以上推荐的10部电影ID获取完成
    #根据推荐电影ID获取电影详细信息存储，并输出
    rec_movies = []
    for i in range(len(rec_moviesID)):
        moviesID = rec_moviesID[i][0]
        moviesID =moviesID[0]-1
        rec_movie = list(movies_dat.iloc[moviesID]) 
        rec_movies.append(rec_movie)     # 电影信息
        #添加电影评分
        col = UR_mat[[moviesID]]
        rec_movie.append( col[(0<col)].mean() )
    rec_movies=rec_movies[:10]
    rec_movies = sorted(rec_movies , key=lambda x: x[3] , reverse=True )
    return   rec_movies
    
        
'''
模块三：老用户推荐模块
        1.根据用户评分矩阵构造相似度矩阵
        2.使用聚类k-mean算法，找出最相似的三个用户，获得用户ID
        方案一：                                                            方案二：
            (使用基于user的CF算法)                                                  （结合使用基于user和基于item的CF算法）         
            3.先从最相似的用户开始，寻找用户未看过的电影                                3.使用基于user的CF算法，从最相似的用户开始找到电影推荐得分最高的7部电影
                电影推荐得分计算公式：score = 用户相似度*0.7+电影评分*0.3    
            4.返回10部电影信息                                                              电影推荐得分计算公式：score = 用户相似度*0.7+电影评分*0.3
                                                                                    4.
                                                                                    5.使用基于item的CF算法，找到评分最高的3部电影
                                                                                    6.返回10部推荐的电影
'''

def login_reco(UR_mat,movies_dat ,loginID):
    UR_mat = preprocessing.scale(UR_mat)
    #构造相似度矩阵
    similar_mat = cosine_similarity(UR_mat , dense_output=True)
    # print(similar_mat)
    silimar_df = pd.DataFrame(similar_mat)
    #截取该用户与其他用户的相似度
    user_similar=silimar_df.iloc[loginID-1]
    #user_similar中的列名称即为用户ID-1 ， 先获取用户ID-1
    user_similar_dict = user_similar.to_dict()
    #与logID用户相似的用户存放字典中，再根据字典中的value值排序取其前三
    user_similar_dict = sorted(user_similar_dict.items() , key=lambda x: x[1] , reverse=True )[1:2] 
    simlar_userID = [user[0]   for user in user_similar_dict  ] #这里获取的只是用户所在行数，真实用户ID还要加一  
    alluser_watch = []
    reco_movies = [ ]
    for ID in simlar_userID:
        user_line=UR_mat[ID]
        for i in range(len(user_line)):
            if user_line[i] != 0:
                alluser_watch.append(i)#这里获取只是电影所在列，真实电影ID还要加一     
    result = Counter(alluser_watch).most_common(10) #[:10]
    rec_moviesID = [x[0]+1 for x in result]
    for id in rec_moviesID:
        reco_movies.append(list(movies_dat.iloc[id]))
    return rec_moviesID ,reco_movies #返回的是真实的电影ID


def show_detail(arr_movie):
    for movie in arr_movie:
        print(movie,'\n')

def accur_recall(userid,moviesID,UR_data):
    accrur_rec =0
    for id in moviesID:
        id -=1
        userid -=1
        if UR_data[userid][id] != 0:
            accrur_rec +=1
    act_watch=0
    for rat in UR_data[userid-1]:
        if rat != 0:
            act_watch +=1
    accruracy = accrur_rec/10
    recall = accrur_rec/act_watch
    return accruracy , recall
        
if __name__ == "__main__":
    #读取文件数据，三表合一，将性别转换为数字，去除 时间戳 、电影名称、邮编属性
    movies_dat = pd.read_csv('F://code//my_project//Advanced_AI//film//movies.CSV' , sep=","  , names=["movieID" , "title" , "genres"])
    ratings_dat = pd.read_csv('F://code//my_project//Advanced_AI//film//ratings.CSV' , sep="," , names=["userID" , "movieID" , "rating" , "timestamp"])
    users_dat = pd.read_csv('F://code//my_project//Advanced_AI//film//users.CSV' , sep="," , names=["userID" , "gender" , "age" , "occupation" , "zip-code"])
    #读取处理好的文件
    RMU_data = pd.read_csv('F://code//my_project//Advanced_AI//film//new_RMU.CSV' , sep=',',names =['userID' , 'movieID' , 'rating ' , 'genres'  , 'gender' , 'age' , 'occupation'] )
    #获取前2000条用户数据
    UR_mat=URmat_fuc(RMU_data)
    
    plt.matshow(UR_mat)
    
    plt.show()
    if input("搜索电影名称或电影类型（y）?：") == 'y':
        name = input("请输入要搜索的内容：")
        rec_movies=search_name(movies_dat  , name ,UR_mat)
        print(rec_movies)
   
    
    elif input("新用户（y）？：") == 'y':
        loginID = input("请输入你的登入ID：")
        reco_movie=new_userreco(users_dat , int(loginID))
        show_detail(reco_movie)
    else:
        loginid=int(input("请输入你的登入ID："))
        moviesID,movies=login_reco(UR_mat,movies_dat ,loginid)
        show_detail(movies)
        acc , reca = accur_recall(loginid , moviesID , UR_mat)
        print(acc , reca)



    

    

     
# new_userreco(users_dat[:2000],2)
# login_reco(UR_mat , 2)
# login_reco(UR_mat,movies_dat,2)
