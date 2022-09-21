import pandas as pd
import matplotlib as mpl
import numpy as np
import scipy as sp

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_800.xls')
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

lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
X_lda_df = pd.DataFrame(X_lda_tn, columns=lda_columns)
X_lda_df['target'] = y

one_df=X_lda_df[X_lda_df['target']==1]
two_df=X_lda_df[X_lda_df['target']==2]
three_df=X_lda_df[X_lda_df['target']==3]
four_df=X_lda_df[X_lda_df['target']==4]
five_df=X_lda_df[X_lda_df['target']==5]
six_df=X_lda_df[X_lda_df['target']==6]

one_df.pop('target')
two_df.pop('target')
three_df.pop('target')
four_df.pop('target')
five_df.pop('target')
six_df.pop('target')

mean1=one_df.mean(0)
mean2=two_df.mean(0)
mean3=three_df.mean(0)
mean4=four_df.mean(0)
mean5=five_df.mean(0)
mean6=six_df.mean(0)

cov1=one_df.cov()
cov2=two_df.cov()
cov3=three_df.cov()
cov4=four_df.cov()
cov5=five_df.cov()
cov6=six_df.cov()

rv1 = sp.stats.multivariate_normal(mean1, cov1)
rv2 = sp.stats.multivariate_normal(mean2, cov2)
rv3 = sp.stats.multivariate_normal(mean3, cov3)
rv4 = sp.stats.multivariate_normal(mean4, cov4)
rv5 = sp.stats.multivariate_normal(mean5, cov5)
rv6 = sp.stats.multivariate_normal(mean6, cov6)

out1=rv1.pdf(X_lda_te)
out2=rv2.pdf(X_lda_te)
out3=rv3.pdf(X_lda_te)
out4=rv4.pdf(X_lda_te)
out5=rv5.pdf(X_lda_te)
out6=rv6.pdf(X_lda_te)

print(out1)
print(out2)
print(out3)
print(out4)
print(out5)
print(out6)

print(y_te)