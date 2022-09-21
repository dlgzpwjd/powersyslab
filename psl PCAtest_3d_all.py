import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800.xls')

X=data.drop(['target', 'type'],axis=1)
y=data.filter(['target'])

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(X)
X_pca=pca.transform(X)

print(X.shape)
print(X_pca.shape)

print(pca.explained_variance_ratio_)

pca_columns=['pca_comp1','pca_comp2','pca_comp3']
X_pca_df=pd.DataFrame(X_pca,columns=pca_columns)
X_pca_df['target']=y
X_pca_df.head(5)

df = X_pca_df
markers=['o','.','X','s','D','v','^','*','s','x','_']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['pca_comp1']
    X2 = df_i['pca_comp2']
    X3 = df_i['pca_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('pca_component1')
ax.set_ylabel('pca_component2')
ax.set_zlabel('pca_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()