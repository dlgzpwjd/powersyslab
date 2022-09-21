import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800.xls')

X=data.drop(['target', 'type'],axis=1)
y=data.filter(['type'])

from sklearn.manifold import TSNE
tsne=TSNE(n_components=3, perplexity=50.0) #n 차원으로 축소
X_tsne=tsne.fit_transform(X,y)

print(X.shape)
print(X_tsne.shape)

tsne_columns=['tsne_comp1','tsne_comp2','tsne_comp3']
X_tsne_df=pd.DataFrame(X_tsne,columns=tsne_columns)
X_tsne_df['target']=y
X_tsne_df.head(0)

df = X_tsne_df
markers=['o','.','X','s','D','v','^']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['tsne_comp1']
    X2 = df_i['tsne_comp2']
    X3 = df_i['tsne_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('tsne_component1')
ax.set_ylabel('tsne_component2')
ax.set_zlabel('tsne_component3')
ax.legend(loc='best')
plt.savefig('TSNE.png')
plt.show()
