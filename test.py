import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

data=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_800.xls')

for i in range(800):
    if data['VA_aph'][i] < 0:
        data['VA_aph'][i]= data['VA_aph'][i]+3.14
    else:
        data['VA_aph'][i]= data['VA_aph'][i]
    if data['VA_bph'][i] < 0:
        data['VA_bph'][i]= data['VA_bph'][i]+3.14
    else:
        data['VA_bph'][i]= data['VA_bph'][i]
    if data['VA_cph'][i] < 0:
        data['VA_aph'][i]= data['VA_cph'][i]+3.14
    else:
        data['VA_aph'][i]= data['VA_cph'][i]
    if data['VB_aph'][i] < 0:
        data['VB_aph'][i]= data['VB_aph'][i]+3.14
    else:
        data['VB_aph'][i]= data['VB_aph'][i]
    if data['VB_bph'][i] < 0:
        data['VB_bph'][i]= data['VB_bph'][i]+3.14
    else:
        data['VB_bph'][i]= data['VB_bph'][i]
    if data['VB_cph'][i] < 0:
        data['VB_cph'][i]= data['VB_cph'][i]+3.14
    else:
        data['VB_cph'][i]= data['VB_cph'][i]
    if data['IA_aph'][i] < 0:
        data['IA_aph'][i]= data['IA_aph'][i]+3.14
    else:
        data['IA_aph'][i]= data['IA_aph'][i]
    if data['IA_bph'][i] < 0:
        data['IA_bph'][i]= data['IA_bph'][i]+3.14
    else:
        data['IA_bph'][i]= data['IA_bph'][i]
    if data['IA_cph'][i] < 0:
        data['IA_cph'][i]= data['IA_cph'][i]+3.14
    else:
        data['IA_cph'][i]= data['IA_cph'][i]
    if data['IB_aph'][i] < 0:
        data['IB_aph'][i]= data['IB_aph'][i]+3.14
    else:
        data['IB_aph'][i]= data['IB_aph'][i]
    if data['IB_bph'][i] < 0:
        data['IB_bph'][i]= data['IB_bph'][i]+3.14
    else:
        data['IB_bph'][i]= data['IB_bph'][i]
    if data['IB_cph'][i] < 0:
        data['IB_cph'][i]= data['IB_cph'][i]+3.14
    else:
        data['IB_cph'][i]= data['IB_cph'][i]
data = pd.DataFrame(data)
data.to_excel(excel_writer='C:/Users/user/Desktop/sample.xlsx')
