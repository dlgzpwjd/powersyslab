import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/user/Desktop/all_feature.xlsx')

X=data.drop(['target', 'type'],axis=1)
y=data.filter(['target'])

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)

print(X.shape)
print(X_pca.shape)

print(pca.get_covariance())

print(pca.singular_values_)
print(pca.components_)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

pca_columns=['pca_comp1','pca_comp2']
X_pca_df=pd.DataFrame(X_pca,columns=pca_columns)
X_pca_df['target']=y
X_pca_df.head()

df = X_pca_df
markers=['o','.','X','s','D','v','^','*','s','x','_']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['pca_comp1']
    X2 = df_i['pca_comp2']
    plt.scatter(X1, X2, 
                marker=mark, 
                label=target_i)

plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.legend()
plt.show()