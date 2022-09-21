#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

#train data
data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_onlygnd_1384.xls')

X_tn=data.drop(['target','type','m','real'], axis=1)
y_tn=data.filter(['target'])
z_tn=data.filter(['type'])

# from sklearn.model_selection import train_test_split
# X_tn, X_te, y_tn, y_te, z_tn, z_te=train_test_split(X, y, z, test_size=0.1,shuffle=True, random_state=1)

X_a_tn=X_tn.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_b_tn=X_tn.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
X_c_tn=X_tn.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row=y_tn.shape[0]
yy_tn=y_tn.to_numpy()
z_tn=z_tn.to_numpy()

yy_a_tn=np.zeros((row,1))
yy_b_tn=np.zeros((row,1))
yy_c_tn=np.zeros((row,1))
    
for i in range(0,row-1):
    if z_tn[i]==1:
        yy_a_tn[i]=yy_tn[i]
        yy_b_tn[i]=0
        yy_c_tn[i]=0
    elif z_tn[i]==2:
        yy_a_tn[i]=0
        yy_b_tn[i]=yy_tn[i]
        yy_c_tn[i]=0
    elif z_tn[i]==3:
        yy_a_tn[i]=0
        yy_b_tn[i]=0
        yy_c_tn[i]=yy_tn[i]
    else:
        yy_a_tn[i]=0
        yy_b_tn[i]=0
        yy_c_tn[i]=0
        
yy_a_tn=pd.DataFrame(yy_a_tn)
yy_b_tn=pd.DataFrame(yy_b_tn)
yy_c_tn=pd.DataFrame(yy_c_tn)

#test data
#X_a_te=X_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
#X_b_te=X_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
#X_c_te=X_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

# row_te=y_te.shape[0]
# yy_te=y_te.to_numpy()
# z_te=z_te.to_numpy()

# yy_a_te=np.zeros((row,1))
# yy_b_te=np.zeros((row,1))
# yy_c_te=np.zeros((row,1))
    
# for i in range(0,row_te-1):
#     if z_te[i]==1:
#         yy_a_te[i]=yy_te[i]
#         yy_b_te[i]=0
#         yy_c_te[i]=0
#     elif z_te[i]==2:
#         yy_a_te[i]=0
#         yy_b_te[i]=yy_te[i]
#         yy_c_te[i]=0
#     elif z_te[i]==3:
#         yy_a_te[i]=0
#         yy_b_te[i]=0
#         yy_c_te[i]=yy_te[i]
#     else:
#         yy_a_te[i]=0
#         yy_b_te[i]=0
#         yy_c_te[i]=0
        
# yy_a_te=pd.DataFrame(yy_a_te)
# yy_b_te=pd.DataFrame(yy_b_te)
# yy_c_te=pd.DataFrame(yy_c_te)

#LDA_tn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components = 3)

lda1 = lda.fit(X_a_tn,yy_a_tn)
lda2 = lda.fit(X_b_tn,yy_b_tn)
lda3 = lda.fit(X_c_tn,yy_c_tn)

Xa_lda_tn=lda1.transform(X_a_tn)
Xb_lda_tn=lda2.transform(X_b_tn)
Xc_lda_tn=lda3.transform(X_c_tn)

Xa_lda_tn_df = pd.DataFrame(Xa_lda_tn)
Xb_lda_tn_df = pd.DataFrame(Xb_lda_tn)
Xc_lda_tn_df = pd.DataFrame(Xc_lda_tn)

#LDA_te
# Xa_lda_te=lda1.transform(X_a_te)
# Xb_lda_te=lda2.transform(X_b_te)
# Xc_lda_te=lda3.transform(X_c_te)

# Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
# Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
# Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

keras.utils.set_random_seed(1)

model = Sequential()

from keras.utils import to_categorical
yyy_a_tn=to_categorical(yy_a_tn)
yyy_b_tn=to_categorical(yy_b_tn)
yyy_c_tn=to_categorical(yy_c_tn)

# yyy_a_te=to_categorical(yy_a_te)
# yyy_b_te=to_categorical(yy_b_te)
# yyy_c_te=to_categorical(yy_c_te)

#testtt
data_a=pd.read_excel('C:/Users/user/Desktop/psl/psl_gndnnormal_131.xlsx')
X_test=data_a.drop(['target','type','m'],axis=1)
y_test=data_a.filter(['target'])
z_test=data_a.filter(['type'])

X_a_test=X_test.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_b_test=X_test.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
X_c_test=X_test.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])

row_test=y_test.shape[0]
yy_test=y_test.to_numpy()
z_test=z_test.to_numpy()

yy_a_test=np.zeros((row_test,1))
yy_b_test=np.zeros((row_test,1))
yy_c_test=np.zeros((row_test,1))
    
for i in range(0,row_test-1):
    if z_test[i]==1:
        yy_a_test[i]=yy_test[i]
        yy_b_test[i]=0
        yy_c_test[i]=0
    elif z_test[i]==2:
        yy_a_test[i]=0
        yy_b_test[i]=yy_test[i]
        yy_c_test[i]=0
    elif z_test[i]==3:
        yy_a_test[i]=0
        yy_b_test[i]=0
        yy_c_test[i]=yy_test[i]
    else:
        yy_a_test[i]=0
        yy_b_test[i]=0
        yy_c_test[i]=0
        
yy_a_test=pd.DataFrame(yy_a_test)
yy_b_test=pd.DataFrame(yy_b_test)
yy_c_test=pd.DataFrame(yy_c_test)

Xa_lda_test=lda1.transform(X_a_test)
Xb_lda_test=lda2.transform(X_b_test)
Xc_lda_test=lda3.transform(X_c_test)

Xa_lda_test_df = pd.DataFrame(Xa_lda_test)
Xb_lda_test_df = pd.DataFrame(Xb_lda_test)
Xc_lda_test_df = pd.DataFrame(Xc_lda_test)

yyy_a_test=to_categorical(yy_a_test)
yyy_b_test=to_categorical(yy_b_test)
yyy_c_test=to_categorical(yy_c_test)

model = load_model("hj_learning_a.h5")

model.evaluate(Xa_lda_test, yyy_a_test, batch_size = 1)
#modela.predict(Xa_lda_test, batch_size = 1)

# modelb = load_model("hj_learning_b.h5")

# modelb.evaluate(Xb_lda_test, yyy_b_test, batch_size = 1)
# #modelb.predict(Xb_lda_test, batch_size = 1)

# modelc = load_model("hj_learning_c.h5")

# modelc.evaluate(Xc_lda_test, yyy_c_test, batch_size = 1)
#modelc.predict(Xc_lda_test, batch_size = 1)