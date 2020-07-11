#importing database
import numpy as np 
import pandas as pd 
dataset = pd.read_csv('BC_data.csv')
X = dataset.iloc[:,2:31].values
Y = dataset.iloc[:,1].values


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
#from 80% data it will train itself and rest 20% will serve to test the trained machine
# 4 new variable are made - X_train,X_test,Y_train,Y_test and a=the original variables X and Y will remain uneffected


#scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#it helps to compare the values of columns with large difference by scaling them to some same range
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

#predict
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, Y_pred)


