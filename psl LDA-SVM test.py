import pandas as pd
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import GridSearchCV

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_800.xls')

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

from sklearn import svm 
clf_svm_lr = svm.SVC(kernel='rbf', gamma=1)
clf_svm_lr.fit(X_lda_tn, y_tn)

pred_svm = clf_svm_lr.predict(X_lda_te)
print(pred_svm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, pred_svm)
print(accuracy)


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_svm)
print(conf_matrix)

from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_svm)
print(class_report)