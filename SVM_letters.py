import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\SVM\\letters.csv")

df.isnull().sum()

X=df.iloc[:,1:17]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

help(SVC)

model=SVC(kernel='linear')
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

###########################################################################################
help(SVC)
model2=SVC(kernel='rbf')
model2.fit(X_train,y_train)

y_pred2=model2.predict(X_test)



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred2)
confusion_matrix(y_test,y_pred2)
classification_report(y_test,y_pred2)
#######################################################################################################

model3=SVC(kernel='poly')
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred3)
confusion_matrix(y_test,y_pred3)
classification_report(y_test,y_pred3)


