import pandas as pd
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')
X=data.drop(['target','type'],axis=1)
y=data.filter(['target'])

from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,test_size=0.3,shuffle=True, random_state=100)

print(X_tn.shape)
print(X_te.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_tn,y_tn)
X_lda_tn=lda.transform(X_tn)
X_lda_te=lda.transform(X_te)

print(X_lda_tn.shape)
print(lda.explained_variance_ratio_)

clf_logi_l2 =  LogisticRegression(penalty='l2', multi_class='multinomial', C=10000, solver='newton-cg')
clf_logi_l2.fit(X_lda_tn, y_tn)

# 로지스틱 회귀분석 모형(L2 제약식 적용) 추정 계수
print(clf_logi_l2.coef_)
print(clf_logi_l2.intercept_)

# 예측
pred_logistic = clf_logi_l2.predict(X_lda_te)
print(pred_logistic)

# 확률값으로 예측
pred_proba = clf_logi_l2.predict_proba(X_lda_te)
print(pred_proba)

# confusion matrix 확인 
conf_matrix = confusion_matrix(y_te, pred_logistic)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_logistic)
print(class_report)