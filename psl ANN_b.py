#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'],axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

#from sklearn.model_selection import train_test_split
#X_tn, X_te, y_tn, y_te, z_tn, z_te=train_test_split(X,y,z,test_size=0.2,shuffle=True, random_state=1)

X_b_tn=X_tn.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_tn[i]=0
    elif z_tn[i]==2:
        yy_tn[i]=yy_tn[i]
    elif z_tn[i]==3:
        yy_tn[i]=0
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)
print(yy_tn)

#test data
# X_b_te=X_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])

# row_te=y_te.shape[0]
# yy_te=y_te.to_numpy()
# z_te=z_te.to_numpy()
    
# for i in range(0,row_te-1):
#     if z_te[i]==1:
#         yy_te[i]=0
#     elif z_te[i]==2:
#         yy_te[i]=yy_te[i]
#     elif z_te[i]==3:
#         yy_te[i]=0
#     else:
#         yy_te[i]=0
        
# yy_te=pd.DataFrame(yy_te)
# print(yy_te)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda2 = lda.fit(X_b_tn,yy_tn)

Xb_lda_tn=lda2.transform(X_b_tn)

print(Xb_lda_tn.shape)

print(lda.explained_variance_ratio_)

Xb_lda_tn_df = pd.DataFrame(Xb_lda_tn)

#LDA_te
# Xb_lda_te=lda2.transform(X_b_te)

# print(Xb_lda_te.shape)

# Xb_lda_te_df = pd.DataFrame(Xb_lda_te)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

keras.utils.set_random_seed(1)

model = Sequential()

from keras.utils import to_categorical
yyy_tn=to_categorical(yy_tn)
#yyy_te=to_categorical(yy_te)

# Adding the input layer and first hidden layer
model.add(Dense(36, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
model.add(Dense(27, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer
model.add(Dense(7, kernel_initializer='he_normal', activation = 'softmax'))

model.summary()

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

epoch=150
results=model.fit(Xb_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

result=np.array(results.history['val_accuracy'])
acc=np.array(results.history['accuracy'])

for i in range(epoch):
    if result[i] > result[i-1]:
        model.save('C:/Python/hj_learning_b.h5')
    else:
        pass
    
print('Max of acc: ',max(results.history['accuracy']))
print('Max of Val_acc: ',max(results.history['val_accuracy']))

plt.plot(results.history['val_accuracy'])
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_acc', 'acc'], loc='upper left')
plt.savefig('test.png')
plt.show()

from keras.models import load_model
model = load_model("hj_learning_b.h5")

# model.evaluate(Xb_lda_te, yyy_te, batch_size = 1)
# model.predict(Xb_lda_te, batch_size = 1)

data_a=pd.read_excel('C:/Users/user/Desktop/psl/psl_gndnnormal_131.xlsx')
X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

X_a=X.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

yy_a=np.zeros((row,1))
    
for i in range(0,row-1):
    if z[i]==1:
        yy_a[i]=0
    elif z[i]==2:
        yy_a[i]=yy[i]
    elif z[i]==3:
        yy_a[i]=0
    else:
        yy_a[i]=0
        
yy_a=pd.DataFrame(yy_a)

Xa_lda=lda2.transform(X_a)

print(Xa_lda.shape)

Xa_lda_df = pd.DataFrame(Xa_lda)

yyy_a=to_categorical(yy_a)

from keras.models import load_model
modela = load_model("hj_learning_b.h5")

modela.evaluate(Xa_lda, yyy_a, batch_size = 1)
modela.predict(Xa_lda, batch_size = 1)