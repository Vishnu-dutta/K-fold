from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower names'] = df.target.apply(lambda x: iris.target_names[x])
# print(df)

print(np.average((cross_val_score(LogisticRegression(), iris.data, iris.target))))
print(np.average((cross_val_score(SVC(), iris.data, iris.target))))
print(np.average((cross_val_score(RandomForestClassifier(), iris.data, iris.target))))
print(np.average((cross_val_score(DecisionTreeClassifier(), iris.data, iris.target))))
