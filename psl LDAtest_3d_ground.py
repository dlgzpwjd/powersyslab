import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
lda=LinearDiscriminantAnalysis(n_components=3)
# lda.fit(X_a_te,yy_te)
lda.fit(X_b_te,yy_te)
# lda.fit(X_c_te,yy_te)

# Xa_lda_te=lda.transform(X_a_te)
Xb_lda_te=lda.transform(X_b_te)
# Xc_lda_te=lda.transform(X_c_te)

# print(Xa_lda_te.shape)
print(Xb_lda_te.shape)
# print(Xc_lda_te.shape)

# Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
# Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

print(lda.explained_variance_ratio_)

lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
X_lda_df = pd.DataFrame(Xb_lda_te, columns=lda_columns)
X_lda_df['target'] = yy_te

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
plt.title('LDA_b')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()