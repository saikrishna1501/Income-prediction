# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:29:42 2019

@author: sai
"""

import numpy as np #numpy library contains mathematical tools
import matplotlib.pyplot as plt #used for plotting
import pandas as pd #pandas is used to import dataset

dataset = pd.read_csv('adult.csv')   # read csv file

x_train = dataset.iloc[: , 0:-1].values  # read features [all rows,all columns except the last]
y_train = dataset.iloc[: , -1].values #read the dependent coloum [all rows,only last column]

np.set_printoptions(threshold = np.nan) #dont care about this
from sklearn.impute import SimpleImputer #simple imputer is used to fill the missing values

simpleimputer = SimpleImputer(' ?' , "most_frequent")  #most frequent strategy used to fill the missing values for categorical data
simpleimputer = simpleimputer.fit(x_train[:,[1,3,5,6,7,8,9,13]]) 

x_train[:,[1,3,5,6,7,8,9,13]] = simpleimputer.transform(x_train[:,[1,3,5,6,7,8,9,13]])
#fit and transform

simpleimputer = SimpleImputer(np.nan,"mean") #fill the missing values using mean stategy np.nan represent empty values

simpleimputer = simpleimputer.fit(x_train[:,[0,2,4,10,11,12]])

x_train[:,[0,2,4,10,11,12]] = simpleimputer.transform(x_train[:,[0,2,4,10,11,12]])
#fit and transform

"""
from sklearn.model_selection import train_test_split 
#split the data into test and train where test size = 25% and train size = 80%
x_train1 , x_test1 , y_train1 ,y_test1 = train_test_split(x,y,test_size = 0.2,random_state = 0)
"""




from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder is used to convert categorical data into numeric data
from sklearn.compose import ColumnTransformer
for i in [1,3,5,6,7,8,9,13] : #here i is all categorical column indices
    labelencoder_x = LabelEncoder()
    labelencoder_x = labelencoder_x.fit(x_train[:,i])
    
    x_train[:,i]= labelencoder_x.transform(x_train[:,i]) # [all rows,i th column]
#fit and transform
"""
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(categories = "auto"), # The transformer class
         [1,3,5,6,7,8,9,13]           # The column(s) to be applied on.
         )
    ]
    ,remainder='passthrough' # not mentioned columns are allowed to passthrough if 'drop' is mentioned instead of
    #of 'passthrough' then not mentioned coloumns will be droped
)
x = transformer.fit_transform(x)
"""


labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)  #convert categorical data to numeric data

y_train=y_train.astype('int') 

from sklearn.preprocessing import StandardScaler #standadization (x - mean(x))/standard deviation(x)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit and transform
#x_test = sc_x.transform(x_test)  #strictly only transform






dataset1 = pd.read_csv('adulttest123.csv')   # read csv file

x_test = dataset1.iloc[1: , 0:-1].values  # read features [all rows,all columns except the last]
y_test = dataset1.iloc[1: , -1].values #read the dependent coloum [all rows,only last column]

np.set_printoptions(threshold = np.nan) #dont care about this
from sklearn.impute import SimpleImputer #simple imputer is used to fill the missing values

simpleimputer = SimpleImputer(' ?' , "most_frequent")  #most frequent strategy used to fill the missing values for categorical data
simpleimputer = simpleimputer.fit(x_test[:,[1,3,5,6,7,8,9,13]]) 

x_test[:,[1,3,5,6,7,8,9,13]] = simpleimputer.transform(x_test[:,[1,3,5,6,7,8,9,13]])
#fit and transform

simpleimputer = SimpleImputer(np.nan,"mean") #fill the missing values using mean stategy np.nan represent empty values

simpleimputer = simpleimputer.fit(x_test[:,[0,2,4,10,11,12]])

x_test[:,[0,2,4,10,11,12]] = simpleimputer.transform(x_test[:,[0,2,4,10,11,12]])
#fit and transform

"""
from sklearn.model_selection import train_test_split 
#split the data into test and train where test size = 25% and train size = 80%
x_train1 , x_test1 , y_train1 ,y_test1 = train_test_split(x,y,test_size = 0.2,random_state = 0)
"""




from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder is used to convert categorical data into numeric data
from sklearn.compose import ColumnTransformer
for i in [1,3,5,6,7,8,9,13] : #here i is all categorical column indices
    #labelencoder_x = LabelEncoder()
    labelencoder_x = labelencoder_x.fit(x_test[:,i])
    
    x_test[:,i]= labelencoder_x.transform(x_test[:,i]) # [all rows,i th column]
#fit and transform
"""
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(categories = "auto"), # The transformer class
         [1,3,5,6,7,8,9,13]           # The column(s) to be applied on.
         )
    ]
    ,remainder='passthrough' # not mentioned columns are allowed to passthrough if 'drop' is mentioned instead of
    #of 'passthrough' then not mentioned coloumns will be droped
)
x = transformer.fit_transform(x)
"""


#labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)  #convert categorical data to numeric data

y_test=y_test.astype('int')  

x_test = sc_x.transform(x_test)  #strictly only transform







"""from sklearn.preprocessing import StandardScaler
#perform standardadization on the test and train sample (x - mean) / sd
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(categories = "auto"), # The transformer class
         []           # The column(s) to be applied on.
         )
    ]
    ,remainder='passthrough' # not mentioned columns are allowed to passthrough if 'drop' is mentioned instead of
    #of 'passthrough' then not mentioned coloumns will be droped
)
x = transformer.fit_transform(x)"""


from sklearn.tree import DecisionTreeClassifier 

classifier = DecisionTreeClassifier(criterion = 'entropy')  #id3 classifier

classifier.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred = classifier.predict(x_test)  #predict the test set results

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred) #making confusion matrix

#accuracy calculations for id3 algorithm
id3accuracy = (cm[0][0] + cm[1][1]) * 100/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

print('\ntotal accuracy of id3 algorithm : ')
print(id3accuracy)

"""
greaterid3accuracy = (cm[1][1])/(cm[1][0] + cm[1][1])
print('\naccuracy of a person having salary greater than 50k')
print(greaterid3accuracy)

lowerid3accuracy = (cm[0][0])/(cm[0][0] + cm[0][1])
print('\naccuracy of a person having salary lower than 50k')
print(lowerid3accuracy)
print('\n\n\n')
"""
 #uncomment this to print accuracies of salary greater and lower than 50k for id3
#naive bayes
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred = classifier.predict(x_test)  #predict the test set results

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,y_pred) #making confusion matrix

#accuracy calculations for Naive bayes algorithm
bayesaccuracy = (cm1[0][0] + cm1[1][1]) * 100/(cm1[0][0] + cm1[0][1] + cm1[1][0] + cm1[1][1])

print('\ntotal accuracy of Naive bayes algorithm : ')
print(bayesaccuracy)
print('\n\n\n')

"""
greaterbayesaccuracy = (cm1[1][1])/(cm1[1][0] + cm1[1][1])
print('\naccuracy of a person having salary greater than 50k')
print(greaterbayesaccuracy)

lowerbayesaccuracy = (cm1[0][0])/(cm1[0][0] + cm1[0][1])
print('\naccuracy of a person having salary lower than 50k')
print(lowerbayesaccuracy)
""" #uncomment this to print accuracies of salary greater and lower than 50k for NB

"""
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred1 = classifier.predict(x_test)  #predict the test set results

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,y_pred1) #making confusion matrix

#accuracy calculations for id3 algorithm
bayesaccuracy = (cm1[0][0] + cm1[1][1]) * 100/(cm1[0][0] + cm1[0][1] + cm1[1][0] + cm1[1][1])

print('\ntotal accuracy of Naive bayes algorithm : ')
print(bayesaccuracy)
print('\n\n\n')
"""


