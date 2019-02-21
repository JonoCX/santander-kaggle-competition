import xgboost as xgb
import pandas as pd 
import numpy as np 
import util

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

from imblearn import over_sampling
from imblearn.over_sampling import SMOTE

rand_state = np.random.RandomState(42)

data = pd.read_csv('../data/train.csv')

y = data['target']
X = data.drop(['ID_code', 'target'], axis=1)

X, y = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = split(
   X, y, test_size=0.25, random_state=rand_state)


params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [2, 3, 4, 5],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}

model = xgb.XGBClassifier(objective='binary:logistic', silent=True, nthread=1)

kfold = StratifiedKFold(n_splits=5, random_state=rand_state)

search = RandomizedSearchCV(
    model,
    param_distributions=params, 
    scoring='roc_auc',
    n_jobs=4,
    cv=kfold.split(X, y),
    verbose=5, 
    random_state=rand_state
)

start_time = util.timer(None)
search.fit(X, y)
util.timer(start_time)

print('Best Parameters: ', search.best_params_)
print('Best Score: ', search.best_score_)

joblib.dump(search.best_estimator_, 'xgb-model.pkl', compress=1)

search_results = pd.DataFrame(search.cv_results_)
search_results.to_csv('../xgb-boost-param-search-res.csv', index=False)

test_data = pd.read_csv('../data/test.csv')
test_data['target'] = search.predict(test_data[:, test_data.columns != 'ID_code'])
test_data.to_csv('../submission.csv', columns=['ID_code', 'target'])
