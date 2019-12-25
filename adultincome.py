# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:34:21 2019

@author: gizem.senol
"""

import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

scaler = preprocessing.MinMaxScaler()


def k_fold_cross_validation(x, y, estimator):
    success = cross_val_score(estimator = estimator, X=x, y=y , cv = 10)
    return(success.mean())
    
def comparison_values(cm):
    
    accuracy = ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][0]+cm[0][1]+cm[1][1]))
    recall = (cm[0][0]/(cm[0][0]+cm[1][0]))
    specifity =  (cm[1][1]/(cm[1][1]+cm[0][1]))
    precision = (cm[0][0]/(cm[0][0]+cm[0][1]))
    F1_score = (2*recall*precision)/(recall+precision)
    
    Algoritmaların_degerleri={'accuracy':accuracy,
                              'recall': recall,
                              'Specifity': specifity,
                              'Precision':precision,
                              'F1-score':F1_score}
    
    return accuracy, recall, specifity, precision, F1_score ,Algoritmaların_degerleri


def accuracy_(cm):
    accuracy = (cm[0][0]+cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    return accuracy


dataset = pd.read_csv("adult.csv")
describe = dataset.describe()


#missing valueların 
dataset = dataset[dataset != '?']
missing_values_count = dataset.isnull().sum()
dataset = dataset.dropna()


dataset = dataset[dataset['native-country'] == 'United-States']
dataset = dataset.drop(['native-country'], axis=1)

dataset['income'].value_counts().plot(kind='pie')
plt.show()


#datanın dengeli hale getirilmesi
data1 = dataset[dataset['income']=='<=50K']
print("<=50K olanlar-data1:"+ str(data1.shape))
data2 = dataset[dataset['income']=='>50K']
print(">50K olanlar-data2:"+ str(data2.shape))
data = data2.append(data1[:10448])
print("Son veriseti :"+ str(data.shape))


data = data[data['age']<65]



#görselleştirme....
gorsel_data = data

sns.distplot(gorsel_data['age'], hist = False)
plt.show()

gorsel_data['workclass'].value_counts().plot(kind='pie')
plt.show()

sns.distplot(gorsel_data['fnlwgt'], hist = False)
plt.show()

sns.countplot(y="education", data=gorsel_data)
plt.show()

sns.distplot(gorsel_data['educational-num'], hist = False)
plt.show()

sns.countplot(y="marital-status", data=gorsel_data)
plt.show()

gorsel_data['occupation'].value_counts().plot(kind='pie')
plt.show()

gorsel_data['relationship'].value_counts().plot(kind='pie')
plt.show()

sns.countplot(y="race", data=gorsel_data)
plt.show()

gorsel_data['gender'].value_counts().plot(kind='pie')
plt.show()


sns.distplot(gorsel_data['capital-gain'],kde = False)
plt.show()

sns.distplot(gorsel_data['capital-loss'],kde = False)
plt.show()

sns.distplot(gorsel_data['hours-per-week'], hist = False)
plt.show()

gorsel_data['income'].value_counts().plot(kind='pie')
plt.show()
#.....

data['income'] = data['income'].map({'>50K':1,'<=50K':0 });
data['relationship'] = data['relationship'].map({'Husband':'Husband-Wife','Wife':'Husband-Wife','Not-in-family':'Not-in-family', 'Own-child':'Own-child','Unmarried':'Unmarried','Other-relative':'Other' });

#görselleştirme
gorsel_data1 = gorsel_data[gorsel_data['income']==0]
gorsel_data2 = gorsel_data[gorsel_data['income']==1]
sns.distplot(gorsel_data1['age'], hist = False,color = 'green',label = 'geliri 50K altında olanlar')
sns.distplot(gorsel_data2['age'], hist = False,color = 'yellow', label = 'geliri 50K üzerinde olanlar')
plt.show()


#görselleştirme
sns.barplot(x = 'race', y = 'hours-per-week',  data = gorsel_data)
sns.barplot(x = 'race', y = 'income',  data = gorsel_data)
plt.show()


#görselleştirme
sns.barplot(x = 'gender', y = 'income', data = gorsel_data)
sns.barplot(x = 'gender', y = 'hours-per-week',  data = gorsel_data)
plt.show()

#görselleştirme
sns.barplot(x = 'relationship', y = 'hours-per-week',  data = gorsel_data)
sns.barplot(x = 'relationship', y = 'income',  data = gorsel_data)
plt.show()

#görselleştirme
sns.barplot(x = 'educational-num', y = 'hours-per-week',  data = gorsel_data)
plt.show()

#encode dataset
le = preprocessing.LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])
data['occupation'] = le.fit_transform(data['occupation'])
data['education'] = le.fit_transform(data['education'])
data['workclass'] = le.fit_transform(data['workclass'])

#min-max scale
    #1.education
data['education'] = scaler.fit_transform(np.array(data[['education']]).reshape(-1,1))
    #2. occupation
data['occupation'] = scaler.fit_transform(np.array(data[['occupation']]).reshape(-1,1))
    #3.workclass
data['workclass'] = scaler.fit_transform(np.array(data[['workclass']]).reshape(-1,1))
    #4.fnlwgt
data['fnlwgt'] = scaler.fit_transform(np.array(data[['fnlwgt']]).reshape(-1,1))

#korelasyon analizi
corr = data.corr()
sns.heatmap(corr)
plt.show()

#OHE
data = pd.concat([data.iloc[:,:8],pd.get_dummies(data['race']),data.iloc[:,9:]],axis = 1)
data = pd.concat([data.iloc[:,:7],pd.get_dummies(data['relationship']),data.iloc[:,8:]],axis = 1)
data = pd.concat([data.iloc[:,:5],pd.get_dummies(data['marital-status']),data.iloc[:,6:]],axis = 1)

#korelasyon analizi
corr = data.corr()
sns.heatmap(corr)
plt.show()
data = data.drop(['Married-civ-spouse'], axis = 1)

#train test split
X = data.iloc[:,:-1]
Y = data.iloc[:,-1:]
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
accuracy_of_train = {}


#Algoritmaların denenip karşılaştırılırsa:
algoritmaların_k_cross_accuracy_değeri = {}
algoritmaların_accuracy_değeri = {}
algoritmaların_recall_değeri = {}
algoritmaların_specifity_değeri = {}
algoritmaların_precision_değeri = {}
algoritmaların_F1_score_değeri = {}


 # 1. KNN    
from sklearn.neighbors import KNeighborsClassifier  
knn = KNeighborsClassifier(n_neighbors=7) #n sayısı 5 ve 9 olduğunda accuracy 0.80 oluyor bu nedenle optimum değer 7.
knn.fit(x_train,y_train)
knn_y_pred = knn.predict(x_test)  
#confusion matrix of KNN 
cm_knn = confusion_matrix(y_test,knn_y_pred)
accuracy, recall, specifity, precision, F1_score, knn_degerleri = comparison_values(cm_knn)
k_cross_accuracy = k_fold_cross_validation(X, Y, knn)
algoritmaların_k_cross_accuracy_değeri['k_cross_Accuracy of KNN'] = k_cross_accuracy 
algoritmaların_accuracy_değeri['Accuracy of KNN'] = accuracy 
algoritmaların_recall_değeri['Recall of KNN'] = recall
algoritmaların_specifity_değeri['Specifity of KNN'] = specifity
algoritmaların_precision_değeri['Precision of KNN'] = precision
algoritmaların_F1_score_değeri['F1-score of KNN'] = F1_score
knn_degerleri['k cross validation accuracy'] = k_cross_accuracy

#accuracy of train:
KNN_train_pred = knn.predict(x_train) 
cm_KNN_train = confusion_matrix(y_train,KNN_train_pred)   
accuracy_train = accuracy_(cm_KNN_train)
accuracy_of_train['train accuracy of KNN'] = accuracy_train

    
# 2. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)    
gnb_y_pred = gnb.predict(x_test)
#confusion matrix of NaiveBayes
cm_gnb= confusion_matrix(y_test,gnb_y_pred)
accuracy, recall, specifity, precision, F1_score, naive_bayes_degerleri = comparison_values(cm_gnb)
k_cross_accuracy = k_fold_cross_validation(X, Y, gnb)
algoritmaların_k_cross_accuracy_değeri['k_cross_Accuracy of Naive Bayes'] = k_cross_accuracy 
algoritmaların_accuracy_değeri['Accuracy of Naive Bayes'] = accuracy 
algoritmaların_recall_değeri['Recall of Naive Bayes'] = recall
algoritmaların_specifity_değeri['Specifity of Naive Bayes'] = specifity
algoritmaların_precision_değeri['Precision of Naive Bayes'] = precision
algoritmaların_F1_score_değeri['F1-score of Naive Bayes'] = F1_score
naive_bayes_degerleri['k cross validation accuracy'] = k_cross_accuracy

#accuracy of train:
gnb_train_pred = gnb.predict(x_train) 
cm_gnb_train = confusion_matrix(y_train,gnb_train_pred)   
accuracy_train = accuracy_(cm_gnb_train)
accuracy_of_train['train accuracy of Naive Bayes'] = accuracy_train
    

# 3. Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dtc = base_estimator = BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes = 20))
dtc.fit(x_train,y_train)
dtc_y_pred = dtc.predict(x_test)
#confusion matrix of DecisionTreeClassifier
cm_dtc = confusion_matrix(y_test,dtc_y_pred)
accuracy, recall, specifity, precision, F1_score, karar_agacı_degerleri = comparison_values(cm_dtc)
k_cross_accuracy = k_fold_cross_validation(X, Y, dtc)
algoritmaların_k_cross_accuracy_değeri['k_cross_Accuracy of Decision tree'] = k_cross_accuracy 
algoritmaların_accuracy_değeri['Accuracy of Decision tree'] = accuracy 
algoritmaların_recall_değeri['Recall of Decision tree'] = recall
algoritmaların_specifity_değeri['Specifity of Decision tree'] = specifity
algoritmaların_precision_değeri['Precision of Decision tree'] = precision
algoritmaların_F1_score_değeri['F1-score of Decision tree'] = F1_score
karar_agacı_degerleri['k cross validation accuracy'] = k_cross_accuracy

#accuracy of train:
dtc_train_pred = dtc.predict(x_train) 
cm_dtc_train = confusion_matrix(y_train,dtc_train_pred)   
accuracy_train = accuracy_(cm_dtc_train)
accuracy_of_train['train accuracy of Decision tree'] = accuracy_train


# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(min_samples_leaf = 100)
rfc.fit(x_train,y_train)    
rfc_y_pred = rfc.predict(x_test)
#confusion matrix of RandomForestClassifier
cm_rfc = confusion_matrix(y_test,rfc_y_pred)
accuracy, recall, specifity, precision, F1_score, random_forest_degerleri = comparison_values(cm_rfc)
k_cross_accuracy = k_fold_cross_validation(X, Y, rfc)
algoritmaların_k_cross_accuracy_değeri['k_cross_Accuracy of Random Forest'] = k_cross_accuracy 
algoritmaların_accuracy_değeri['Accuracy of Random Forest'] = accuracy 
algoritmaların_recall_değeri['Recall of Random Forest'] = recall
algoritmaların_specifity_değeri['Specifity of Random Forest'] = specifity
algoritmaların_precision_değeri['Precision of Random Forest'] = precision
algoritmaların_F1_score_değeri['F1-score of Random Forest'] = F1_score
random_forest_degerleri['k cross validation accuracy'] = k_cross_accuracy

#accuracy of train:
rfc_train_pred = rfc.predict(x_train) 
cm_rfc_train = confusion_matrix(y_train,rfc_train_pred)   
accuracy_train = accuracy_(cm_rfc_train)
accuracy_of_train['Train accuracy of Random Forest'] = accuracy


#ensemble learning: 

from sklearn.ensemble import VotingClassifier
ens = VotingClassifier(estimators=[('knn', knn), ('gnb', gnb), ('dtc', dtc), ('rfc',rfc)])
ens.fit(x_train, y_train)
ens_y_pred = ens.predict(x_test)
#confusion matrix of RandomForestClassifier
cm_ens = confusion_matrix(y_test,rfc_y_pred)
accuracy, recall, specifity, precision, F1_score, ensemble_degerleri = comparison_values(cm_ens)
k_cross_accuracy = k_fold_cross_validation(X, Y, rfc)
algoritmaların_k_cross_accuracy_değeri['k_cross_Accuracy of Ensemble'] = k_cross_accuracy 
algoritmaların_accuracy_değeri['Accuracy of Ensemble'] = accuracy 
algoritmaların_recall_değeri['Recall of Ensemble'] = recall
algoritmaların_specifity_değeri['Specifity of Ensemble'] = specifity
algoritmaların_precision_değeri['Precision of Ensemble'] = precision
algoritmaların_F1_score_değeri['F1-score of Ensemble'] = F1_score
ensemble_degerleri['k cross validation accuracy'] = k_cross_accuracy

#accuracy of train:
ens_train_pred = ens.predict(x_train) 
cm_ens_train = confusion_matrix(y_train,ens_train_pred)   
accuracy_train = accuracy_(cm_ens_train)
accuracy_of_train['Train accuracy of Ensemble'] = accuracy