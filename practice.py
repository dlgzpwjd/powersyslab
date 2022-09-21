import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'], axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

X_a_tn=X_tn.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_tn[i]=0
    elif z_tn[i]==2:
        yy_tn[i]=0
    elif z_tn[i]==3:
        yy_tn[i]=yy_tn[i]
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda1 = lda.fit(X_a_tn,yy_tn)

Xa_lda_tn=lda1.transform(X_a_tn)

Xa_lda_tn_df = pd.DataFrame(Xa_lda_tn)

print(lda.explained_variance_ratio_)

lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
X_lda_df = pd.DataFrame(Xa_lda_tn, columns=lda_columns)
X_lda_df['target'] = yy_tn

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = X_lda_df
markers=['o','.','X','s','D','v','^']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['lda_comp1']
    X2 = df_i['lda_comp2']
    X3 = df_i['lda_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
plt.title('LDA_c')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()