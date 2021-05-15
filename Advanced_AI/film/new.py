from surprise import SVD
from surprise import Dataset, Reader
# from surprise import print_perf
from surprise.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import os
from sklearn import preprocessing
# 指定文件所在路径
file_path = os.path.expanduser('ratings.csv')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating', sep=',')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
# data = preprocessing.scale(data)
algo = SVD()
# 在数据集上测试一下效果
# perf = cross_validate(algo, data, scoring=["accuracy","recall"], cv=3)
perf = cross_val_score(algo , data , scoring="accuracy , recall",cv=1000)
#输出结果
print(perf)