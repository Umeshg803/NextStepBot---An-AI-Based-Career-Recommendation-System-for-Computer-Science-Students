#--------------------Importing Required Libraries/Modules-----------------------#
import numpy as np
import pandas as pd

dataset = pd.read_csv("roo_data.csv")

#---------Testing by displaying whether data is loaded properly or not-----------#
data = dataset.iloc[:,:-1].values
label = dataset.iloc[:,-1].values
len(data[0])
# dataset.iloc[:,14:38]
# dataset.iloc[:,14:38]

#--------------- Lable  Encoding-----------#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

#---------------conversion of all categorial column values to vector/numerical--------#
for i in range(14,38):
    data[:,i] = labelencoder.fit_transform(data[:,i])  
# data[:5]
# data[:5,14:]

#--------------normalizing the non-categorial column values---------#
from sklearn.preprocessing import Normalizer

data1=data[:,:14]
normalized_data = Normalizer().fit_transform(data1)
# print(normalized_data.shape)
data2=data[:,14:]
# print(data2.shape)
df1 = np.append(normalized_data,data2,axis=1)
# print(df1.shape)

#--------------------------Adding Headers-----------------------#
X1 = pd.DataFrame(df1,columns=['Acedamic percentage in Operating Systems', 'percentage in Algorithms',
    'Percentage in Programming Concepts',
    'Percentage in Software Engineering', 'Percentage in Computer Networks',
    'Percentage in Electronics Subjects',
    'Percentage in Computer Architecture', 'Percentage in Mathematics',
    'Percentage in Communication skills', 'Hours working per day',
    'Logical quotient rating', 'hackathons', 'coding skills rating',
    'public speaking points', 'can work long time before system?',
    'self-learning capability?', 'Extra-courses did', 'certifications',
    'workshops', 'talenttests taken?', 'olympiads',
    'reading and writing skills', 'memory capability score',
    'Interested subjects', 'interested career area ', 'Job/Higher Studies?',
    'Type of company want to settle in?',
    'Taken inputs from seniors or elders', 'interested in games',
    'Interested Type of Books', 'Salary Range Expected',
    'In a Realtionship?', 'Gentle or Tuff behaviour?',
    'Management or Technical', 'Salary/work', 'hard/smart worker',
    'worked in teams ever?', 'Introvert'])
# X1.head()

#------------------Encoding Final Output column Values------------#
label = labelencoder.fit_transform(label)
# print(len(label))
y=pd.DataFrame(label,columns=["Suggested Job Role"])
# y.head()
# print(y)
# print(label)

#------------------Training and testing with Decision Tree----------------#

from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=10) 

#-----------------decision tree-----------------------#
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = clf.predict(X_test)
y_pred
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*100)

#---------------svm------------------------#
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)  
svm_y_pred = clf.predict(X_test)

svm_cm = confusion_matrix(y_test,svm_y_pred)
svm_accuracy = accuracy_score(y_test,svm_y_pred)

print("confusion matrics=",svm_cm)
print("  ")
print("accuracy=",svm_accuracy*100)


#------------------xgboost--------------#
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=10) 
X_train.shape

#------------converting values of training and testing data into int64 datatype-------#
X_train=pd.to_numeric(X_train.values.flatten())
X_train=X_train.reshape((14000,38))

#-------------importing and defining xgboost functions-----#
from xgboost import XGBClassifier

model = XGBClassifier()
#-----------training and testing with xg boost------#
model.fit(X_train, y_train)
xgb_y_pred = clf.predict(X_test)

xgb_cm = confusion_matrix(y_test,xgb_y_pred)
xgb_accuracy = accuracy_score(y_test,xgb_y_pred)

print("confusion matrics=",xgb_cm)
print("  ")
print("accuracy=",xgb_accuracy*100)


import os
#-------------------Saving the  model------------------------#
import pickle

if not os.path.exists('./newmodels'): # create models directory
    os.mkdir('newmodels')

# Save Models
pickle.dump(model, open('newmodels/svc_model.h5', 'wb')) # SVC Model