import xgboost as xgb
import pandas as pd 
import numpy as np 
import util

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold 

rand_state = np.random.RandomState(42)

data = pd.read_csv('../data/train.csv')

y = data['target']
X = data.drop(['ID_code', 'target'], axis=1)

#X_train, X_test, y_train, y_test = split(
#    X, y, test_size=0.25, stratify=y, random_state=rand_state)

# 3*5*3*3*4=540
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [2, 3, 4, 5]
}

model = xgb.XGBClassifier(
    learning_rate=0.02, n_estimators=200, 
    objective='binary:logistic', silent=True, nthread=1)

folds = 3
param_combinations = 5

kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)

search = RandomizedSearchCV(
    model, param_distributions=params, 
    n_iter=param_combinations, scoring='roc_auc',
    n_jobs=4, cv=kfold.split(X, y), 
    verbose=4, random_state=rand_state
)

start_time = util.timer(None)
search.fit(X, y)
util.timer(start_time)

results = pd.DataFrame(search.cv_results_)
results.to_csv('xgb-boost-param-search-res.csv', index=False)

test_data = pd.read_csv('../data/test.csv')
test_data['target'] = search.predict(test_data[:, test_data.columns != 'ID_code'])
test_data.to_csv('../data/submission.csv', columns=['ID_code', 'target'])

