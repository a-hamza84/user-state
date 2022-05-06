#!/usr/bin/python3

import pandas as pd
import numpy as n
TData=pd.read_csv("data.csv")
TData.columns=['RNum','Distance_Covered','Average_Acc_x','Average_Acc_y','Average_Acc_z','Average_Grav_x','Average_Grav_y','Average_Grav_z','Binary_Steps','State']
TData=TData.sample(frac=1).reset_index(drop=True)
div=int(len(TData)*0.7)
TData_train=TData[0:div]
TData_test=TData[div:len(TData)]
classes=['vehicle','transport']

### Naive Bayes
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
GNB=GaussianNB()
GNB.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=GNB.predict(TData_test.iloc[:,1:9])

#> Accuracy Mean squared
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

#>Confusion Matrix
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
print("Done Training")
import pickle
print(GNB)
model_path_filename = 'model_pickle_NB.pkl'
# Open the file to save as pkl file
model_pkl = open(model_path_filename, 'wb')
pickle.dump(GNB, model_pkl)
model_pkl.close()
