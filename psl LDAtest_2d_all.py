
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data_te=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')
X_a_te=data_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_b_te=data_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
X_c_te=data_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])
y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])

row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()
    
for i in range(0,row_te-1):
    if z_te[i]==1:
        yy_te[i]=0
    elif z_te[i]==2:
        yy_te[i]=yy_te[i]
    elif z_te[i]==3:
        yy_te[i]=0
    else:
        yy_te[i]=0
        
yy_te=pd.DataFrame(yy_te)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_b_te,yy_te)
X_lda=lda.transform(X_b_te)

print(X_b_te.shape)
print(X_lda.shape)

print(lda.intercept_)
print(lda.coef_)

lda_columns = ['lda_comp1', 'lda_comp2']
X_lda_df = pd.DataFrame(X_lda, columns=lda_columns)
X_lda_df['target'] = yy_te
X_lda_df.head(5)

import matplotlib.pyplot as plt
df = X_lda_df
markers=['o','.','X','s','D','v','^','*','s','x','_']

for i, mark in enumerate(markers):
    X_i = df[df['target']== i]
    target_i = i
    X1 = X_i['lda_comp1']
    X2 = X_i['lda_comp2']
    plt.scatter(X1, X2, 
                marker=mark, 
                label=target_i)
plt.xlabel('lda_component1')
plt.ylabel('lda_component2')
plt.legend()
plt.show()