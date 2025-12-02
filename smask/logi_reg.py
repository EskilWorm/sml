import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt





#, usecols=["temp","dew","precip","visibility","increase_stock"]

mp = {"low_bike_demand":-1, "high_bike_demand":1}
df = pd.read_csv("training_data_ht2025.csv")

dfTrain = df[:1200]
dfHoldOut = df[1200:]


l = lambda x : 0 if x == 1  else 1

arr = np.array(dfTrain).T
arr2 = np.array(dfHoldOut).T

X = np.array(arr[:-1].T, dtype="float64")


y2 = arr2[-1].T
y3 = np.zeros(len(arr.T))

for i in range(len(y3)):
    y3[i] = mp[arr[-1][i]]

learner = skl_lm.LogisticRegression()
learner.fit(X,y3)
pred = learner.predict(arr2[:-1].T)
actual = np.array([1 if x == "high_bike_demand" else -1 for x in y2])




t = pred*actual
for i in range(400):
    t[i] = l(t[i])
print(np.mean(t))


