import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800.xls')

X=data.drop(['target','type'],axis=1)
y=data.filter(['target'])

from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(X)
X_pca=pca.transform(X)

print(X.shape)
print(X_pca.shape)
print(pca.explained_variance_ratio_)

print(np.sum(pca.explained_variance_ratio_[0:10]))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_pca,y)
X_lda=lda.transform(X_pca)

print(X_lda.shape)

print(lda.explained_variance_ratio_)

lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
X_lda_df = pd.DataFrame(X_lda, columns=lda_columns)
X_lda_df['target'] = y

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = X_lda_df
markers=['o','.','X','s','D','v','^','*','s','x','_']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['lda_comp1']
    X2 = df_i['lda_comp2']
    X3 = df_i['lda_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('pl_component1')
ax.set_ylabel('pl_component2')
ax.set_zlabel('pl_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()