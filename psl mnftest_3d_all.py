import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800_line.xls')

for i in range(384):
    if data['VA_aph'][i] < 0:
        data['VA_aph'][i]= data['VA_aph'][i]+3.14
    else:
        data['VA_aph'][i]= data['VA_aph'][i]
    if data['VA_bph'][i] < 0:
        data['VA_bph'][i]= data['VA_bph'][i]+3.14
    else:
        data['VA_bph'][i]= data['VA_bph'][i]
    if data['VA_cph'][i] < 0:
        data['VA_aph'][i]= data['VA_cph'][i]+3.14
    else:
        data['VA_aph'][i]= data['VA_cph'][i]
    if data['VB_aph'][i] < 0:
        data['VB_aph'][i]= data['VB_aph'][i]+3.14
    else:
        data['VB_aph'][i]= data['VB_aph'][i]
    if data['VB_bph'][i] < 0:
        data['VB_bph'][i]= data['VB_bph'][i]+3.14
    else:
        data['VB_bph'][i]= data['VB_bph'][i]
    if data['VB_cph'][i] < 0:
        data['VB_cph'][i]= data['VB_cph'][i]+3.14
    else:
        data['VB_cph'][i]= data['VB_cph'][i]
    if data['IA_aph'][i] < 0:
        data['IA_aph'][i]= data['IA_aph'][i]+3.14
    else:
        data['IA_aph'][i]= data['IA_aph'][i]
    if data['IA_bph'][i] < 0:
        data['IA_bph'][i]= data['IA_bph'][i]+3.14
    else:
        data['IA_bph'][i]= data['IA_bph'][i]
    if data['IA_cph'][i] < 0:
        data['IA_cph'][i]= data['IA_cph'][i]+3.14
    else:
        data['IA_cph'][i]= data['IA_cph'][i]
    if data['IB_aph'][i] < 0:
        data['IB_aph'][i]= data['IB_aph'][i]+3.14
    else:
        data['IB_aph'][i]= data['IB_aph'][i]
    if data['IB_bph'][i] < 0:
        data['IB_bph'][i]= data['IB_bph'][i]+3.14
    else:
        data['IB_bph'][i]= data['IB_bph'][i]
    if data['IB_cph'][i] < 0:
        data['IB_cph'][i]= data['IB_cph'][i]+3.14
    else:
        data['IB_cph'][i]= data['IB_cph'][i]
        

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])

nmf = NMF(n_components=3, random_state=0)
nmf.fit(X)
X_nmf = nmf.transform(X)

print(X.shape)
print(X_nmf.shape)

nmf_columns = ['nmf_comp1', 'nmf_comp2','nmf_comp3']
X_nmf_df = pd.DataFrame(X_nmf, columns=nmf_columns)
X_nmf_df['target'] = y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = X_nmf_df
markers=['o','.','X','s','D','v','^']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['nmf_comp1']
    X2 = df_i['nmf_comp2']
    X3 = df_i['nmf_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('nmf_component1')
ax.set_ylabel('nmf_component2')
ax.set_zlabel('nmf_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()