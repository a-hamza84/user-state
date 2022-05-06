# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:50:03 2018

@author: user
"""

import pandas as pd
import numpy as n
TData=pd.read_csv("E:/ITU/Final Year Project/Test data/data.csv")
TData.columns=['RNum','Distance_Covered','Average_Acc_x','Average_Acc_y','Average_Acc_z','Average_Grav_x','Average_Grav_y','Average_Grav_z','Binary_Steps','State']
TData=TData.sample(frac=1).reset_index(drop=True)
div=int(len(TData)*0.7)
TData_train=TData[0:div]
TData_test=TData[div:len(TData)]
classes=['vehicle','transport']
#####Logistic Regression between distance covered and status
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error # This is the best way to test Regression models
Log_Reg=LogisticRegression()
Log_Reg.fit(TData_train['Distance_Covered'].reshape(-1,1),TData_train['State'].reshape(-1,1))
predicted=Log_Reg.predict(TData_test['Distance_Covered'].reshape(-1,1))
#> Accuracy:
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

##### Plotting graph

import matplotlib.pyplot as plt #to import matplotlib you need tkinter as well

plt.scatter(TData_train['Distance_Covered'].reshape(-1,1),TData_train['State'].reshape(-1,1),color='black')
plt.plot(TData_test['Distance_Covered'].reshape(-1,1), predicted, color='blue',linewidth=2)
plt.xlabel("Distance covered within 30 sec")
plt.ylabel("State of the user recorded")
plt.show()

#####Logistic Regression between multivariate
from sklearn.metrics import confusion_matrix
Log_Reg=LogisticRegression()
Log_Reg.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=Log_Reg.predict(TData_test.iloc[:,1:9])
#> Accuracy Mean Squared error:
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

###decision tree
from sklearn import tree
Tree=tree.DecisionTreeClassifier()
Tree.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=Tree.predict(TData_test.iloc[:,1:9])
Tree.score(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
tree.export_graphviz(Tree,out_file='E:/ITU/Final Year Project/plots/tree.dot')
#> Showing tree as image
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz  
import pydotplus

dot_data = StringIO()
export_graphviz(Tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
#> accuracy mean squared error
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### Naive Bayes
from sklearn.naive_bayes import GaussianNB
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
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

### As GNB has best accuracy level the 

### kNN
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
KNN.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=KNN.predict(TData_test.iloc[:,1:9])

#> Accuracy Mean squared
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

#>Confusion Matrix
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#SVM
from sklearn import svm
SVM=svm.SVC()
SVM.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=SVM.predict(TData_test.iloc[:,1:9])

#> Accuracy Mean squared
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

#>Confusion Matrix
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Clustering
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=2,random_state=0)
cluster.fit(TData_train.iloc[:,1:9])
predicted=cluster.predict(TData_test.iloc[:,1:9])

#> Accuracy Mean squared
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

#>Confusion Matrix
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#ANN
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(3,3), random_state=1)

clf.fit(TData_train.iloc[:,1:9],TData_train.iloc[:,9])
predicted=clf.predict(TData_test.iloc[:,1:9])

#> Accuracy Mean squared
print("The error: ",mean_squared_error(TData_test['State'],predicted),"%")
print("Accuracy: ",1-mean_squared_error(TData_test['State'],predicted),"%")

#>Confusion Matrix
#> Accuracy Confusion matrix
Con_Mat=confusion_matrix(TData_test['State'],predicted)
print(Con_Mat)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Con_Mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()