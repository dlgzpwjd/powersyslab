#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

test = input("train : 0 / test : 1 >> ")

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'], axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

X_a_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_tn[i]=yy_tn[i]
    elif z_tn[i]==2:
        yy_tn[i]=0
    elif z_tn[i]==3:
        yy_tn[i]=0
    else:
        yy_tn[i]=0
        
yy_tn=pd.DataFrame(yy_tn)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)
lda1 = lda.fit(X_a_tn,yy_tn)

Xa_lda_tn=lda1.transform(X_a_tn)

Xa_lda_tn_df = pd.DataFrame(Xa_lda_tn)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical

if test == 1:
    keras.utils.set_random_seed(1)

    model = Sequential()

    
    yyy_tn=to_categorical(yy_tn)
    # yyy_te=to_categorical(yy_te)

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
    results=model.fit(Xa_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

    result=np.array(results.history['val_accuracy'])
    acc=np.array(results.history['accuracy'])

    for i in range(epoch):
        if result[i] > result[i-1]:
            model.save('C:/Python/hj_learning_a.h5')
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

data_a=pd.read_excel('C:/Users/user/Desktop/psl/psl_gndnnormal_131.xlsx')
X=data_a.drop(['target','type','m'],axis=1)
y=data_a.filter(['target'])
z=data_a.filter(['type'])

X_a=X.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

yy_a=np.zeros((row,1))
    
for i in range(0,row-1):
    if z[i]==1:
        yy_a[i]=yy[i]
    elif z[i]==2:
        yy_a[i]=0
    elif z[i]==3:
        yy_a[i]=0
    else:
        yy_a[i]=0
        
yy_a=pd.DataFrame(yy_a)

Xa_lda=lda1.transform(X_a)

Xa_lda_df = pd.DataFrame(Xa_lda)

yyy_a=to_categorical(yy_a)

modela = load_model("hj_learning_a.h5")

#modela.evaluate(Xa_lda, yyy_a, batch_size = 1)
pred_a = modela.predict(Xa_lda, batch_size = 1)

################################################ b ################################################

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'],axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

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

#LDA_tn
lda=LinearDiscriminantAnalysis(n_components = 3)
lda2 = lda.fit(X_b_tn,yy_tn)

Xb_lda_tn=lda2.transform(X_b_tn)

Xb_lda_tn_df = pd.DataFrame(Xb_lda_tn)

#ANN

if test == 1:

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

data_a=pd.read_excel('C:/Users/user/Desktop/psl/psl_gndnnormal_131.xlsx')
X=data_a.drop(['target','type','m'],axis=1)
y=data_a.filter(['target'])
z=data_a.filter(['type'])

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

Xa_lda_df = pd.DataFrame(Xa_lda)

yyy_a=to_categorical(yy_a)

modelb = load_model("hj_learning_b.h5")

#modelb.evaluate(Xa_lda, yyy_a, batch_size = 1)
pred_b = modelb.predict(Xa_lda, batch_size = 1)

################################################ c ################################################

#LDA

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'],axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

X_c_tn=X_tn.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_tn[i]=0
    elif z_tn[i]==2:
        yy_tn[i]=0
    elif z_tn[i]==3:
        yy_tn[i]=yy_tn[i]
    else:
        yy_tn[i]=0

yy_tn=pd.DataFrame(yy_tn)

#LDA_tn
lda=LinearDiscriminantAnalysis(n_components = 3)

lda3 = lda.fit(X_c_tn,yy_tn)

Xc_lda_tn=lda3.transform(X_c_tn)

Xc_lda_tn_df = pd.DataFrame(Xc_lda_tn)

#ANN

if test == 1:

    keras.utils.set_random_seed(1)

    model = Sequential()

    yyy_tn=to_categorical(yy_tn)
    #yyy_te=to_categorical(yy_te)

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
    results=model.fit(Xc_lda_tn, yyy_tn, batch_size = 10, epochs = epoch, validation_split = 0.2)

    result=np.array(results.history['val_accuracy'])
    acc=np.array(results.history['accuracy'])

    for i in range(epoch):
        if result[i] > result[i-1]:
            model.save('C:/Python/hj_learning_c.h5')
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

data_a=pd.read_excel('C:/Users/user/Desktop/psl/psl_gndnnormal_131.xlsx')
X=data_a.drop(['target','type','m'],axis=1)
y=data_a.filter(['target'])
z=data_a.filter(['type'])

X_a=X.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

yy_a=np.zeros((row,1))
    
for i in range(0,row-1):
    if z[i]==1:
        yy_a[i]=0
    elif z[i]==2:
        yy_a[i]=0
    elif z[i]==3:
        yy_a[i]=yy[i]
    else:
        yy_a[i]=0
        
yy_a=pd.DataFrame(yy_a)

Xa_lda=lda3.transform(X_a)

Xa_lda_df = pd.DataFrame(Xa_lda)

yyy_a=to_categorical(yy_a)

modelc = load_model("hj_learning_c.h5")

#modelc.evaluate(Xa_lda, yyy_a, batch_size = 1)
pred_c = modelc.predict(Xa_lda, batch_size = 1)

# 비교할 원본 테스트 데이터
con_test = np.zeros((row, 3))

for i in range(0, row-1):
    if z[i] == 1:
        con_test[i, 0] = yy[i]
    elif z[i] == 2:
        con_test[i, 1] = yy[i]
    elif z[i] == 3:
        con_test[i, 2] = yy[i]
       
# 비교할 테스트 데이터 
pred_a = np.reshape(np.argmax(pred_a, axis=1), (-1, 1))
pred_b = np.reshape(np.argmax(pred_b, axis=1), (-1, 1))
pred_c = np.reshape(np.argmax(pred_c, axis=1), (-1, 1))

pred_test = np.concatenate((pred_a, pred_b, pred_c), axis=1)

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(pred_test)

last = (con_test == pred_test)
print(last)

accuracy = np.sum(last) / (last.shape[0] * last.shape[1]) * 100

print("Accuracy : %.4f %%" %(accuracy))

pred_test1 = pd.DataFrame(pred_test)
con_test1 = pd.DataFrame(con_test)
last1 = pd.DataFrame(last)
xlsx = 'C:/Users/user/Desktop/result1.xlsx'
with pd.ExcelWriter(xlsx) as writer:
    pred_test1.to_excel(writer, sheet_name = 'pred_test1')
    con_test1.to_excel(writer, sheet_name = 'con_test')
    last1.to_excel(writer, sheet_name = 'last')