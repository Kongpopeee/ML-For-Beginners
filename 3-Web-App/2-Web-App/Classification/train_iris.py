import pandas as pd 
import numpy as np 

df = pd.read_csv('iris.csv') 
df.head() 


# Renaming the target column into numbers to aid training of the model 
df['variety']= df['variety'].map({'Setosa':0, 'Versicolor':1, 'Virginica':2}) 

# splitting the data into the columns which need to be trained(X) and the target column(y) 
X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 

# splitting data into training and testing data with 30 % of data as testing data respectively 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# importing the random forest classifier model and training it on the dataset 
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier() 
classifier.fit(X_train, y_train) 

# predicting on the test dataset 
y_pred = classifier.predict(X_test) 

# finding out the accuracy 
from sklearn.metrics import accuracy_score 
score = accuracy_score(y_test, y_pred) 

# pickling the model 
import pickle 
pickle_out = open("classifier.pkl", "wb") 
pickle.dump(classifier, pickle_out) 
pickle_out.close()
