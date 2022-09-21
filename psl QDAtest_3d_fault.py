import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800.xls')

X=data.drop(['target','type'],axis=1)
y=data.filter(['target'])

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda=QuadraticDiscriminantAnalysis()
X_qda=qda.fit(X,y)

qda_columns = ['qda_comp1', 'qda_comp2','qda_comp3']
X_qda_df = pd.DataFrame(X_qda, columns=qda_columns)
X_qda_df['target'] = y

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = X_qda_df
markers=['o','.','X','s','D','v','^']

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['qda_comp1']
    X2 = df_i['qda_comp2']
    X3 = df_i['qda_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('qda_component1')
ax.set_ylabel('qda_component2')
ax.set_zlabel('qda_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()