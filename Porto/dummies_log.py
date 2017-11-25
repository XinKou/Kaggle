#Dummies for categorical attributes and log transformation for all attributes, just replace your file path, X_train_log and X_test_log are transformed array for fit and predict

import pandas as pd
import numpy as np

#feature engineering for X_train, return X_train_log
#replace with your train_path
train_path = '~/kaggle/porto_seguro_safe_driver/datasets/train.csv'
train = pd.read_csv(train_path)
train.drop('id', axis = 1, inplace = True)
X_train = train.drop('target', axis=1)
y_train = train['target']
#replace -1 with 0
replace_dict = {-1:0}
X_train = X_train.replace(replace_dict)
columns = list(X_train.columns)
#find categorical attributes and numerical attributes
cat_attribs = []
num_attribs = []
for attribs in columns:
    if 'cat' in attribs:
        cat_attribs.append(attribs)
    else: num_attribs.append(attribs)
#onehotencoder for categorical attributes
from sklearn.preprocessing import OneHotEncoder
train_cat = OneHotEncoder().fit_transform(X_train[cat_attribs])
train_num = np.array(X_train[num_attribs])
import scipy
X_train_concat = scipy.sparse.hstack((train_cat,train_num)).A
#np.log1p for all attributes
X_train_log = np.log1p(X_train_concat)


#feature engineering for X_test, return X_test_log
#replace with your test path
raw_test_path = '~/kaggle/porto_seguro_safe_driver/datasets/test.csv'
test = pd.read_csv(raw_test_path)
test.drop('id', axis = 1, inplace = True)
X_test = test.replace(replace_dict)
from sklearn.preprocessing import OneHotEncoder
test_cat = OneHotEncoder().fit(X_train[cat_attribs]).transform(X_test[cat_attribs])
test_num = np.array(X_test[num_attribs])
import scipy
X_test_concat = scipy.sparse.hstack((test_cat,test_num)).A
X_test_log = np.log1p(X_test_concat)
