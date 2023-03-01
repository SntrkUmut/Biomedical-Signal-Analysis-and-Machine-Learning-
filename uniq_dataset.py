# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:40:58 2023

@author: umuts
"""


import matplotlib.pylab as plt
import os

import pandas as pd
import numpy as np
import scipy

### Machine Learning
from sklearn.model_selection import train_test_split
##Feature Engineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pickle


dd=pd.read_csv(r'C:\Users\umuts\Desktop\EMG.csv')

all_data=dd.iloc[:10000000,:]

def extractStatisticalfeatures(x):
   
    fstd=np.std(x)
    fmin=np.min(x)
    fkurtosis=scipy.stats.kurtosis(x)
    zero_crosses = np.nonzero(np.diff(x > 0))[0]
    fzero=zero_crosses.size/len(x)
    frms = np.sqrt(np.mean(np.square(x)))
    return fstd,fkurtosis,fzero,frms,fmin

#Features
fstd=[]
fkurtosis=[]
fzero=[]
frms=[]
fmin=[]
flabel=[]
   
ch1mean=[]
ch2mean=[]
ch3mean=[]
ch4mean=[]
ch5mean=[]
ch6mean=[]
ch7mean=[]
ch8mean=[]
    
fpercent=[]
flabel2=[]
winhop = 50
winsize = 750
for i in range(0,len(all_data),winhop):
    
    selmat=all_data.iloc[i:i+winsize, 1:9].to_numpy().flatten()     
    sta,kur,zer,rms,minu = extractStatisticalfeatures(selmat)
    fstd.append(sta)
    fkurtosis.append(kur)
    fzero.append(zer)
    frms.append(rms)
    fmin.append(minu)   
        
    ch1mean.append(all_data.iloc[i:i+winsize,1].mean())
    ch2mean.append(all_data.iloc[i:i+winsize,2].mean())
    ch3mean.append(all_data.iloc[i:i+winsize,3].mean())
    ch4mean.append(all_data.iloc[i:i+winsize,4].mean())
    ch5mean.append(all_data.iloc[i:i+winsize,5].mean())
    ch6mean.append(all_data.iloc[i:i+winsize,6].mean())
    ch7mean.append(all_data.iloc[i:i+winsize,7].mean())
    ch8mean.append(all_data.iloc[i:i+winsize,8].mean())
        
    bincountlist=np.bincount(all_data.iloc[i:i+winsize,-1].to_numpy(dtype='int64'))
    most_frequent_class=bincountlist.argmax()
    flabel.append(most_frequent_class)
        
    percentage_most_frequent=bincountlist[most_frequent_class]/len(all_data.iloc[i:i+winsize,-1].to_numpy(dtype='int64'))
    fpercent.append(percentage_most_frequent)
        
    if percentage_most_frequent==1.0:
        most_frequent_class2=most_frequent_class
    else:
        bincountlist[most_frequent_class]= 0
        most_frequent_class2=bincountlist.argmax()
            
    flabel2.append(most_frequent_class2)
        
rdf = pd.DataFrame(
       {'ch1mean': ch1mean,
        'ch2mean': ch2mean,
        'ch3mean': ch3mean,
        'ch4mean': ch4mean,
        'ch5mean': ch5mean,
        'ch6mean': ch6mean,
        'ch7mean': ch7mean,
        'ch8mean': ch8mean,
        'std': fstd,
        'min': fmin,
        'kurtosis': fkurtosis,
        'zerocross':fzero,
        'rms':frms,
        'label':flabel,
        'percent':fpercent,
        '2ndlabel':flabel2
        
    })

X=rdf.iloc[:,0:13]
y=rdf.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

ran = RandomForestClassifier(n_estimators = 100)

ran.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = ran.predict(X_test)
  
# metrics are used to find accuracy or error
from sklearn import metrics  

  

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))



# importing the required modules
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Plot the confusion matrix in graph
cm = confusion_matrix(y_test,y_pred, labels=ran.classes_)
# ploting with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ran.classes_)
disp.plot()
# showing the matrix
plt.show()



from sklearn.metrics import confusion_matrix, classification_report

# View the classification report for test data and predictions
print(classification_report(y_test, y_pred))



# Create Decision Tree classifer
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

Xd=rdf.iloc[:,0:13]
yd=rdf.iloc[:,13]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(Xd, yd, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Xg=rdf.iloc[:,0:13]
yg=rdf.iloc[:,13]


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(Xg, yg, test_size=0.3,random_state=109) # 70% training and 30% test


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import pickle

pickle.dump(ran, open('model.pickle', 'wb'))


