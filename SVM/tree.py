from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score

X,y=datasets.load_breast_cancer(return_X_y=True)
tree = tree.DecisionTreeClassifier()
scores=cross_val_score(tree, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())