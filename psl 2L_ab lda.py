import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_500_2L.xls')

X_tn=data.drop(['target','type','m'], axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

X_ab_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==4:
        yy_tn[i]=yy_tn[i]
    elif z_tn[i]==5:
        yy_tn[i]=0
    elif z_tn[i]==6:
        yy_tn[i]=0
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)
print(yy_tn.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda1 = lda.fit(X_ab_tn,yy_tn)

Xab_lda_tn=lda1.transform(X_ab_tn)

print(Xab_lda_tn.shape)

print(lda.explained_variance_ratio_)


lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
Xab_lda_df = pd.DataFrame(Xab_lda_tn, columns=lda_columns)
Xab_lda_df['target'] = yy_tn

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = Xab_lda_df
markers = ['o','.','X','s','D','v','^']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    type_i = i
    X1 = df_i['lda_comp1']
    X2 = df_i['lda_comp2']
    X3 = df_i['lda_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark,
                label=type_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()