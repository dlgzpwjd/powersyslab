import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 1000 2L.xls')

X_tn=data.drop(['target','type','m'], axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

X_ac_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==4:
        yy_tn[i]=0
    elif z_tn[i]==5:
        yy_tn[i]=0
    elif z_tn[i]==6:
        yy_tn[i]=yy_tn[i]
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)
print(yy_tn.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda1 = lda.fit(X_ac_tn,yy_tn)

Xac_lda_tn=lda1.transform(X_ac_tn)

print(Xac_lda_tn.shape)

print(lda.explained_variance_ratio_)


lda_columns = ['lda_comp1', 'lda_comp2','lda_comp3']
Xac_lda_df = pd.DataFrame(Xac_lda_tn, columns=lda_columns)
Xac_lda_df['target'] = yy_tn

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = Xac_lda_df
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

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical


keras.utils.set_random_seed(1)

model = Sequential()

    
yyy_tn=to_categorical(yy_tn)

# Adding the input layer and first hidden layer
model.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
model.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer
model.add(Dense(7, kernel_initializer='he_normal', activation = 'softmax'))

model.summary()

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

epoch=150
results=model.fit(Xac_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

result=np.array(results.history['val_accuracy'])
acc=np.array(results.history['accuracy'])

for i in range(epoch):
    if result[i] > result[i-1]:
        model.save('C:/Python/hj_learning_ac.h5')
    else:
        pass
        
print('Max of acc: ',max(results.history['accuracy']))
print('Max of Val_acc: ',max(results.history['val_accuracy']))

plt.plot(results.history['val_accuracy'])
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_acc', 'acc'], loc='lower right')
plt.savefig('test.png')
plt.show()

