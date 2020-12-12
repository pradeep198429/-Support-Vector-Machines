import pandas as pd 
import numpy as np 
# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
# for one hot encoding with feature-engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder



train = pd.read_csv("SalaryData_Train(1).csv")
test = pd.read_csv("SalaryData_Test(1).csv")

numerical=['age','educationno','capitalgain','capitalloss']
categorical = ['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']

#train[numerical]
k=train[categorical]

type(k)
for col in k.columns:
    print(col, ': ', len(train[col].unique()), ' labels')
    
 for col in k.columns:
    print(col, ': ', len(test[col].unique()), ' labels')
    
train['native'].value_counts().sort_values(ascending=False).head(40)

test['native'].value_counts().sort_values(ascending=False).head(40)

ohe_enc = OneHotCategoricalEncoder(
    top_categories=14,  # you can change this value to select more or less variables
    # we can select which variables to encode
    variables=['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],
    drop_last=False)

ohe_enc.fit(train)

train = ohe_enc.transform(train)
test = ohe_enc.transform(test)


X_train = train.drop(['Salary'],axis=1)
y_train = train[['Salary']]
X_test = test.drop(['Salary'],axis=1)
y_test = test[['Salary']]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train['Salary'].value_counts()

y_train['Salary'].replace({" <=50K": 0," >50K" : 1},inplace=True)

y_test['Salary'].replace({" <=50K": 0 , " >50K" : 1},inplace = True)

y_train['Salary'].value_counts()

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train,np.array( y_train).ravel())

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
for i in ['linear','rbf']:
    for j in ['scale','auto']:
        for k in [1]:
            classifier = SVC(kernel = i,gamma= j,C=k, random_state = 0)
            classifier.fit(X_train_res, y_train_res)
            y_pred = classifier.predict(X_test)
            accu = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print("accuracy for {} with gamma as {} for c= {} is {}".format(i,j,k,accu))
            print("---"*20)
            print("\n\n")

