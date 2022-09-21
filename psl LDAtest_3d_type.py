import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_excel('C:/Users/user/Desktop/fault_feature_1000.xls')

X=data.drop(['target','type'],axis=1)
y=data.filter(['type'])

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X,y)
X_lda=lda.transform(X)

print(X.shape)
print(X_lda.shape)

print(lda.explained_variance_ratio_)

lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
X_lda_df = pd.DataFrame(X_lda, columns=lda_columns)
X_lda_df['target'] = y

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = X_lda_df
colors = ['g', 'b' ,'r' , 'y']

for i, color in enumerate(colors):
    df_i = df[df['type']== i]
    type_i = i
    X1 = df_i['lda_comp1']
    X2 = df_i['lda_comp2']
    X3 = df_i['lda_comp3']
    ax.scatter(X1, X2, X3,
                c=color, 
                label=type_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()