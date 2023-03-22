from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

'''
Using KFold cross validation
'''
kf = KFold(n_splits=3)
print(kf)

for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)

'''
Using KFold for our digits example
'''


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

'''
Using Cross_val_score function
'''

print(cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=3))
print(cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3))
