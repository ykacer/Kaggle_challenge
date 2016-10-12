# Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

# Kaggle challenge : Allstate Claims Severity

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn import grid_search

from sklearn import linear_model
from sklearn.svm import SVR

print("Formatting data")

name_col = list(pd.read_csv('train.csv',header=-1,encoding='utf-8',nrows=1,low_memory=False).values.flatten()) # read header firstly
type_col = [np.chararray if name[:3]==u'cat' else np.float64 for name in name_col] # get type of each column
name2type_dict = dict(zip(name_col,type_col))

train_data = pd.read_csv('train.csv',header=0,encoding='utf-8',dtype=name2type_dict) # now read data faster thx to dictionary type
data = np.array(train_data);
y = data[:,-1]; # take loss from last column skipping
X = data[:,1:-1]; # take variables from other columns skipping id column

test_data = pd.read_csv('test.csv',header=0,encoding='utf-8',dtype=name2type_dict) # read test data faster thx to dictionary type
data = np.array(test_data);
X_test = data[:,1:];

print ("Encode categorical variables")

# look for categorical variables
isCategorical = np.nonzero([1 if name[:3] == u'cat' else 0 for name in name_col[1:]])[0] # get col indices of categorical variables
print(isCategorical)
for ic in isCategorical:
	label_encoder = LabelEncoder()
	X[:,ic] = label_encoder.fit_transform(X[:,ic])
	label_encoder = LabelEncoder()
        X_test[:,ic] = label_encoder.fit_transform(X_test[:,ic])

categorical_encoder = OneHotEncoder(categorical_features=isCategorical)
categorical_encoder.fit(np.vstack((X,X_test)));
Xencode = categorical_encoder.transform(X);
Xencode_test = categorical_encoder.transform(X_test);

print("Classification bench")
cv = ShuffleSplit(y.size,test_size=0.3) # cross-validation set
results = [];
verbose = 2

print("* Linear Regression")
cl = linear_model.LinearRegression()
param_grid = {'fit_intercept':[True,False]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xencode,y)
ytest = grid.best_estimator_.predict(Xencode_test)
results.append(['Linear Regression',grid.grid_scores_,grid.scorer_,ytest])

print("* Ridge Regression")
cl = linear_model.Ridge()
param_grid = {'fit_intercept':[True,False],'alpha':[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xencode,y)
ytest = grid.best_estimator_.predict(Xencode_test)
results.append(['Ridge Regression',grid.grid_scores_,grid.scorer_,ytest])

print("* Lasso Regression")
cl = linear_model.Lasso()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xencode,y)
ytest = grid.best_estimator_.predict(Xencode_test)
results.append(['Lasso Regression',grid.grid_scores_,grid.scorer_,ytest])

print("* ElasticNet Regression")
cl = linear_model.ElasticNet()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0],'l1_ratio':[0.25,0.5,0.75]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xencode,y)
ytest = grid.best_estimator_.predict(Xencode_test)
results.append(['ElasticNet Regression',grid.grid_scores_,grid.scorer_,ytest])

print("* Ransac Regression")
base_estimator = linear_model.LinearRegression()
cl = linear_model.RANSACRegressor(base_estimator=base_estimator,min_samples=0.8,loss='squared_loss')
cl.fit(Xencode,y)
ytest = cl.estimator_.predict(Xencode_test)
results.append(['Ransac Regression',cl.score(Xencode[cl.inlier_mask_,:],y[cl.inlier_mask_]),None,ytest])

print("* Support Vector Regression")
cl = SVR(kernel='rbf')
param_grid = {'gamma':[0.0,0.1,1.0,10.0,100.0],'C':[0.01,0.1,1.0,10.0,100.0],'epsilon':[0.001,0.01,0.1,1.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xencode,y)
ytest = grid.best_estimator_.predict(Xencode_test)
results.append(['Support Vector Regression',grid.grid_scores_,grid.scorer_,ytest])






