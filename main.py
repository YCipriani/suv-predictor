import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


suv_data = pd.read_csv('Social_Network_Ads.csv')

#iloc function is used as index for pandas dataframe, used for integer based indexing

#for this datset we only care about the age and estimated salary columns to predict the purchase
X= suv_data.iloc[:,[2,3]].values
#target column will be the purchased column
y = suv_data.iloc[:,4].values

#splitting and training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

#in order to scale down input values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

logmodel = LogisticRegression(random_state=10)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

#to calculate accuracy
print('Accuracy score is: ' + str(accuracy_score(y_test, predictions)*100) + '%')