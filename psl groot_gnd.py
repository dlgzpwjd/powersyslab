import pandas as pd
import matplotlib as mpl
import numpy as np
from sklearn import tree 

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_gnd_784.xls')
X=data.drop(['target','type'],axis=1)
y=data.filter(['target'])

from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,test_size=0.3,shuffle=True, random_state=100)

print(X_tn.shape)
print(X_te.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=5)
lda.fit(X_tn,y_tn)
X_lda_tn=lda.transform(X_tn)
X_lda_te=lda.transform(X_te)

print(X_lda_tn.shape)
print(lda.explained_variance_ratio_)

# 의사결정나무 학습
clf_tree = tree.DecisionTreeClassifier(random_state=0)
clf_tree.fit(X_lda_tn, y_tn)

# 예측
pred_tree = clf_tree.predict(X_lda_te)
print(pred_tree)

# f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_te, pred_tree, average='macro')
print(f1)

# confusion matrix 확인 
conf_matrix = confusion_matrix(y_te, pred_tree)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_tree)
print(class_report)