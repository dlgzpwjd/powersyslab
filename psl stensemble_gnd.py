from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib as mpl
import numpy as np

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_gnd_784.xls')
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

# 스태킹 학습
clf1 = svm.SVC(kernel='linear', random_state=1) 
clf2 = GaussianNB()

clf_stkg = StackingClassifier(
            estimators=[
                ('svm', clf1), 
                ('gnb', clf2)
            ],
            final_estimator=LogisticRegression())
clf_stkg.fit(X_lda_tn, y_tn)

# 예측
pred_stkg = clf_stkg.predict(X_lda_te)
print(pred_stkg)

# 정확도
accuracy = accuracy_score(y_te, pred_stkg)
print(accuracy)

# confusion matrix 확인 
conf_matrix = confusion_matrix(y_te, pred_stkg)
print(conf_matrix)

# 분류 레포트 확인
class_report = classification_report(y_te, pred_stkg)
print(class_report)