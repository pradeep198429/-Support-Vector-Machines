import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\SalaryData_Train.csv")
df1=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\SVM\\SalaryData_Test.csv")

trainX=df.iloc[:,1:13]
trainy=df.iloc[:,13]
testX=df1.iloc[:,0:14]
testy=df1.iloc[:,13]

model=SVC(kernel='linear')

model.fit(trainX,trainy)

string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']
from sklearn import preprocessing
for i in string_columns:
    number=preprocessing.LabelEncoder()
    trainX[i]=number.fit_transform(trainX[i])
    testX[i]=number.fit_transform(testX[i])
    
model.fit(trainX,trainy)

y_pred=model.predict(testX)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(testy,y_pred)

accuracy_score(testy,y_pred)
