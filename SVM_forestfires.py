import pandas as pd 
import numpy as np 
# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

dataframe=pd.read_csv("forestfires.csv")

dataframe =dataframe.drop(['month','day'], axis=1)

dataframe["size_category"].replace({"small":0,"large":1} ,inplace=True)

dataframe["size_category"].value_counts()

dataframe.info()
dataframe=dataframe.apply(lambda col:pd.to_numeric(col,errors='coerce'))

X_train, X_test, y_train, y_test = train_test_split(
    dataframe.drop(labels='size_category', axis=1),  # predictors
    dataframe['size_category'],  # target
    test_size=0.2,
    random_state=0)

X_train.shape, X_test.shape

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train,np.array( y_train).ravel())

X_train_res.shape,y_train_res.shape

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

for i in ['linear','rbf']:
    for j in ['scale','auto']:
        for k in [1,10,50,100]:
            classifier = SVC(kernel = i,gamma= j,C=k, random_state = 0)
            classifier.fit(X_train_res, y_train_res)
            y_pred = classifier.predict(X_test)
            accu = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print("accuracy for {} with gamma as {} for c= {} is {}".format(i,j,k,accu))
            print("---"*20)
            print("\n\n")
