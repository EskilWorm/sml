import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



high_low = np.vectorize(lambda x: 1.0 if x == 'high_bike_demand' else -1.0)

df = pd.read_csv("training_data_ht2025.csv")
df['increase_stock'] = high_low(df['increase_stock'])
features = df.drop(columns=['increase_stock','snow','weekday','summertime','dew', 'visibility','holiday'])


features["hour_sin"] = np.sin(2*np.pi*features["hour_of_day"]/24)
features["hour_cos"] = np.cos(2*np.pi*features["hour_of_day"]/24)

features["day_sin"] = np.sin(2*np.pi*features["day_of_week"]/7)
features["day_cos"] = np.cos(2*np.pi*features["day_of_week"]/7)

features["month_sin"] = np.sin(2*np.pi*features["month"]/12)
features["month_cos"] = np.cos(2*np.pi*features["month"]/12)

features = features.drop(columns=["hour_of_day"])
features = features.drop(columns=["day_of_week"])
features = features.drop(columns=["month"])


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, df['increase_stock'], test_size=0.2, random_state=42)

smote = SMOTE()

parameters = {'C':[np.pow(2.0,a) for a in range(-5, 6)],
        'solver':['lbfgs', 'liblinear'],
        'penalty':['l2']}


grid = sklearn.model_selection.GridSearchCV(estimator=sklearn.linear_model.LogisticRegression(max_iter=1000), param_grid=parameters, scoring='f1', cv=10, n_jobs=1)

grid.fit(X_train, Y_train)
cvParams = grid.best_params_
best_model = grid.best_estimator_


Y_predict = best_model.predict(X_test)

test_accuracy = sklearn.metrics.accuracy_score(Y_predict, Y_test)
test_f1 = sklearn.metrics.f1_score(Y_predict, Y_test, pos_label=1)

print(f'parameters:{grid.best_params_}')
print(f'best (f1)score on training data:{grid.best_score_}')
print(f'accuracy:{test_accuracy}')
print(f'f1:{test_f1}')

