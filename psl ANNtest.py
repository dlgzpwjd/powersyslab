#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1184.xls')

X=data.drop(['target','type','m','real'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])

from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te, z_tn, z_te=train_test_split(X,y,z,test_size=0.1,shuffle=True, random_state=1)

X_a_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_b_tn=X_tn.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
X_c_tn=X_tn.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_tn[i]=yy_tn[i]-1
    elif z_tn[i]==2:
        yy_tn[i]=yy_tn[i]+5
    else:
        yy_tn[i]=yy_tn[i]+11
        
yy_tn=pd.DataFrame(yy_tn)

#test data
X_a_te=X_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_b_te=X_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
X_c_te=X_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()
    
for i in range(0,row_te-1):
    if z_te[i]==1:
        yy_te[i]=yy_te[i]-1
    elif z_te[i]==2:
        yy_te[i]=yy_te[i]+5
    else:
        yy_te[i]=yy_te[i]+11
        
yy_te=pd.DataFrame(yy_te)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda1 = lda.fit(X_a_tn,y_tn)
lda2 = lda.fit(X_b_tn,y_tn)
lda3 = lda.fit(X_c_tn,y_tn)

Xa_lda_tn=lda1.transform(X_a_tn)
Xb_lda_tn=lda2.transform(X_b_tn)
Xc_lda_tn=lda3.transform(X_c_tn)

print(Xa_lda_tn.shape)
print(Xb_lda_tn.shape)
print(Xc_lda_tn.shape)

print(lda.explained_variance_ratio_)

Xa_lda_tn_df = pd.DataFrame(Xa_lda_tn)
Xb_lda_tn_df = pd.DataFrame(Xb_lda_tn)
Xc_lda_tn_df = pd.DataFrame(Xc_lda_tn)

X_lda_tn=pd.concat([Xa_lda_tn_df,Xb_lda_tn_df,Xc_lda_tn_df],axis = 1)
print(X_lda_tn.shape)

#LDA_te
Xa_lda_te=lda1.transform(X_a_te)
Xb_lda_te=lda2.transform(X_b_te)
Xc_lda_te=lda3.transform(X_c_te)

print(Xa_lda_te.shape)
print(Xb_lda_te.shape)
print(Xc_lda_te.shape)

Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

X_lda_te=pd.concat([Xa_lda_te_df,Xb_lda_te_df,Xc_lda_te_df], axis = 1)
print(X_lda_te.shape)

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
yyy_te=to_categorical(yy_te)

# Adding the input layer and first hidden layer
model.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 9))
# Adding the second hidden layer
model.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the output layer
model.add(Dense(18, kernel_initializer='he_normal', activation = 'softmax'))

model.summary()

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

epoch=1000
results=model.fit(X_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

result=np.array(results.history['val_accuracy'])
acc=np.array(results.history['accuracy'])

for i in range(epoch):
    if result[i] > result[i-1]:
        model.save('C:/Python/hj_learning.h5')
    else:
        pass

for i in range(epoch):
    if acc[i] < acc[i-1]:
        model.save('C:/Python/hj_learning2.h5')
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
model = load_model("hj_learning.h5")

model.evaluate(X_lda_te,yyy_te, batch_size = 1)
model.predict(X_lda_te,batch_size = 1)

model = load_model("hj_learning2.h5")

model.evaluate(X_lda_te,yyy_te, batch_size = 1)
model.predict(X_lda_te,batch_size = 1)