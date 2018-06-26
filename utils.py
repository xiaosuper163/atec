import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import missingno as msno
import pickle
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from scipy import stats
from scipy.stats import zscore
# from fancyimpute import *

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)

def batch_save(train_x, train_y, valid_x, valid_y, test, postfix):
    train_x.reset_index().to_feather("tmp/train_x_{}".format(postfix))
    train_y.reset_index().to_feather("tmp/train_y_{}".format(postfix))
    valid_x.reset_index().to_feather("tmp/valid_x_{}".format(postfix))
    valid_y.reset_index().to_feather("tmp/valid_y_{}".format(postfix))
    test.reset_index().to_feather("tmp/test_{}".format(postfix))
    
def batch_load(postfix):
    train_x = pd.read_feather("tmp/train_x_{}".format(postfix))
    train_y = pd.read_feather("tmp/train_y_{}".format(postfix))
    valid_x = pd.read_feather("tmp/valid_x_{}".format(postfix))
    valid_y = pd.read_feather("tmp/valid_y_{}".format(postfix))
    return train_x, train_y, valid_x, valid_y

def process_dates(df,date):
    attrs = ['Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    for attr in attrs:
        df[attr] = getattr(df[date].dt, attr.lower())
    return df

def my_roc(y_true, y_prob):
    if isinstance(y_true,pd.core.series.Series):
        y_true = np.array(y_true.tolist())
    if isinstance(y_true,list):
        y_true = np.array(y_true)
    sort_index = np.argsort(y_prob)[::-1]
    y_prob = y_prob[sort_index]
    y_true = y_true[sort_index]
    num_p = y_true.sum()
    num_n = len(y_true) - num_p
    fp = 0
    tp = 0
    fps = []
    tps = []
    prob_prev = -99
    i = 0
    while i < len(y_true):
        if y_prob[i]!=prob_prev:
            fps.append(fp/num_n)
            tps.append(tp/num_p)
            prob_prev=y_prob[i]
        if y_true[i]==1:
            tp+=1
        else:
            fp+=1
        i+=1
    fps.append(fp/num_n)
    tps.append(tp/num_p)
    return np.array(fps), np.array(tps)

def my_score1(ground_truth, predictions): ##Adapted from SKlearn, conservative (actual should be higher)
    fpr,tpr,threhold = roc_curve(ground_truth, predictions) 
    tpr1 = tpr[(fpr>=0.001).argmax()-1]
    tpr2 = tpr[(fpr>=0.005).argmax()-1] 
    tpr3 = tpr[(fpr>=0.01).argmax()-1]
    return 0.4 * tpr1 + 0.3 * tpr2 + 0.3* tpr3

def my_score2(predictions, xtrain): ##Adapted from SKlearn, conservative (actual should be higher)
    ground_truth = xtrain.get_label()
    fpr,tpr = my_roc(ground_truth, predictions)
    tpr1 = tpr[(fpr>=0.001).argmax()-1]
    tpr2 = tpr[(fpr>=0.005).argmax()-1] 
    tpr3 = tpr[(fpr>=0.01).argmax()-1]
    return 'score', 0.4 * tpr1 + 0.3 * tpr2 + 0.3* tpr3, True

def my_score3(predictions, xtrain): ##Adapted from SKlearn, conservative (actual should be higher)
    ground_truth = xtrain.get_label()
    fpr,tpr = my_roc(ground_truth, predictions)
#     plt.scatter(fpr, tpr)
#     plt.show()
    tpr1 = tpr[(fpr>=0.001).argmax()-1]
    tpr2 = tpr[(fpr>=0.005).argmax()-1] 
    tpr3 = tpr[(fpr>=0.01).argmax()-1]
    return 'score', 0.4 * tpr1 + 0.3 * tpr2 + 0.3* tpr3

def get_ratio(predictions, xtrain):
    ratio_predict = (predictions>0.5).sum()/predictions.shape[0]*100
    # ratio_true = xtrain.get_label().sum()/xtrain.get_label().shape[0]*100
    return 'score', ratio_predict

def norm_standardize(df, start=0):
    for col in df.columns[start:]:
#         avg = df[col].mean()
#         std = df[col].std(ddof=0)
#         if std != 0:
#             df[col] = (df[col]-avg)/std
#         else:
#             print(col)
        a = df[col]
        z = a
        z[~np.isnan(a)] = zscore(a[~np.isnan(a)])
        df[col] = z
            
def norm_maxmin(df, start=0):
    for col in df.columns[start:]:
        df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())