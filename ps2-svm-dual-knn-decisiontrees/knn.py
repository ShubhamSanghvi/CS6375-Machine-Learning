# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:23:14 2020

@author: shubh
"""
import pandas as pd
import numpy as np


def knn_classifier(X_test,k,X_train,Y_train,distance_func):

    # apply to each row
    dist = np.apply_along_axis(distance_func, 1, X_train, X_test)
    # print(dist)

    # pick the k-closest
    idx = np.argpartition(dist, k)[:k]
    # print(idx)

    # will count the number of True examples in the idx selected by the selection
    pos = np.sum(Y_train[idx])
    # print(Y_train[idx])
    # print(pos)

    # check if more than half of the k examples selected are positive 
    if pos > (k/2):
      Y_test = True 
    else:
      Y_test = False

    return Y_test

def euclidean(vec1,vec2):
  vec = (vec1 - vec2) ** 2
  ans = np.sum(vec)
  return np.sqrt(ans)

def get_mean_std(in_array):
    return (in_array.mean(),in_array.std())

def normalize_arr(in_array,mean,std):
    return (in_array-mean)/std

# Driver Code
if __name__ == '__main__':
    
    df_train = pd.read_csv('spam_train.data', sep=',',header=None)
    df_dev = pd.read_csv('spam_validation.data', sep=',',header=None)
    df_test = pd.read_csv('spam_test.data', sep=',',header=None)

# prepare the numpy arrays for faster predictions
    X_train = df_train.iloc[:,:-1].values
    Y_train =  df_train.iloc[:,-1].values
    X_dev = df_dev.iloc[:,:-1].values
    Y_dev =  df_dev.iloc[:,-1].values

    X_test = df_test.iloc[:,:-1].to_numpy()
    Y_test =  df_test.iloc[:,-1].to_numpy()

    mean_std = np.apply_along_axis(get_mean_std,axis=0,arr=X_train)
    
    X_train_norm = np.empty_like(X_train)
    for col in range(np.shape(X_train)[1]):
        X_train_norm[:,col] = normalize_arr(X_train[:,col],mean_std[0,col],mean_std[1,col])

    X_dev_norm = np.empty_like(X_dev)
    for col in range(np.shape(X_dev)[1]):
        X_dev_norm[:,col] = normalize_arr(X_dev[:,col],mean_std[0,col],mean_std[1,col])

    X_test_norm = np.empty_like(X_test)
    for col in range(np.shape(X_test)[1]):
        X_test_norm[:,col] = normalize_arr(X_test[:,col],mean_std[0,col],mean_std[1,col])
    
    results = []
    k_values = [1,5,11,15,21]
    
    for k in k_values:
      # apply to each row in our X_dev
      train_pred = np.apply_along_axis(knn_classifier, 1, X_train_norm,k,X_train_norm,Y_train,euclidean)
      train_acc = np.sum(train_pred==Y_train)/np.shape(Y_train)

      val_pred = np.apply_along_axis(knn_classifier, 1, X_dev_norm,k,X_train_norm,Y_train,euclidean)
      val_acc = np.sum(val_pred == Y_dev)/np.shape(Y_dev)

      test_pred = np.apply_along_axis(knn_classifier, 1, X_test_norm,k,X_train_norm,Y_train,euclidean)
      test_acc = np.sum(test_pred == Y_test)/np.shape(Y_test)
      
      results.append([k,train_acc[0],val_acc[0],test_acc[0]])