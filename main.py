import numpy as np
import pandas as pd
import extractor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)

# data extraction
X_suji = extractor.features(13,"data/sujata/")
X_suji= pd.DataFrame(X_suji)
X_suji['Class']=0
print('\nDataset 1 extracted')
X_suji

X_prat = extractor.features(13,"data/pratiksha/")
X_prat= pd.DataFrame(X_prat)
X_prat['Class']=1
print('\nDataset 2 extracted')

X_daksh = extractor.features(13,"data/dakshina/")
X_daksh= pd.DataFrame(X_daksh)
X_daksh['Class']=2
print('\nDataset 3 extracted')

X_saiya = extractor.features(13,"data/sanyogeeta/")
X_saiya= pd.DataFrame(X_saiya)
X_saiya['Class']=3
print('\nDataset 4 extracted')

X_ratnu = extractor.features(13,"data/ratnaparakhi/")
X_ratnu= pd.DataFrame(X_ratnu)
X_ratnu['Class']=4
print('\nDataset 5 extracted')

X_nik = extractor.features(13,"data/nikita/")
X_nik= pd.DataFrame(X_nik)
X_nik['Class']=5
print('\nDataset 6 extracted')

X_ptb = extractor.features(13,"data/pratibha/")
X_ptb= pd.DataFrame(X_ptb)
X_ptb['Class']=6
print('\nDataset 7 extracted')

X_son = extractor.features(13,"data/sonali/")
X_son= pd.DataFrame(X_son)
X_son['Class']=7
print('\nDataset 8 extracted')

X_sim = extractor.features(13,"data/simran/")
X_sim= pd.DataFrame(X_sim)
X_sim['Class']=8
print('\nDataset 9 extracted')

print('\nData Extraction Complete.')

#defining training and testing data
X_suji_train = X_suji[:int(X_suji.shape[0]*0.8)]
X_suji_test = X_suji[int(X_suji.shape[0]*0.8):]
X_prat_train = X_prat[:int(X_prat.shape[0]*0.8)]
X_prat_test = X_prat[int(X_prat.shape[0]*0.8):]

X_daksh_train = X_daksh[:int(X_daksh.shape[0]*0.8)]
X_daksh_test = X_daksh[int(X_daksh.shape[0]*0.8):]

X_saiya_train = X_saiya[:int(X_saiya.shape[0]*0.8)]
X_saiya_test = X_saiya[int(X_saiya.shape[0]*0.8):]

X_ratnu_train = X_ratnu[:int(X_ratnu.shape[0]*0.8)]
X_ratnu_test = X_ratnu[int(X_ratnu.shape[0]*0.8):]

X_nik_train = X_nik[:int(X_nik.shape[0]*0.8)]
X_nik_test = X_nik[int(X_nik.shape[0]*0.8):]

X_ptb_train = X_ptb[:int(X_ptb.shape[0]*0.8)]
X_ptb_test = X_ptb[int(X_ptb.shape[0]*0.8):]
X_son_train = X_son[:int(X_son.shape[0]*0.8)]
X_son_test = X_son[int(X_son.shape[0]*0.8):]

X_sim_train = X_sim[:int(X_sim.shape[0]*0.8)]
X_sim_test = X_sim[int(X_sim.shape[0]*0.8):]

X_train=X_suji_train.append([X_prat_train,X_daksh_train,X_saiya_train,X_ratnu_train,X_nik_train,X_ptb_train,X_son_train, X_sim_train])
X_test=X_suji_test.append([X_prat_test,X_daksh_test,X_saiya_test,X_ratnu_test,X_nik_test,X_ptb_test,X_son_test, X_sim_test])

y_train=X_train[['Class']]
y_test=X_test[['Class']]
X_data=X_train.append(X_test)
X_data = X_data.reset_index(drop=True)
y_data=y_train.append(y_test)
y_data = y_data.reset_index(drop=True)
print('\nPre-processing Done.')

print('\nCount of different classes in Train set:')
print(X_train['Class'].value_counts())

print('\nCount of different classes in Test set:')
print(X_test['Class'].value_counts())

feats=[c for c in X_train.columns if c!='Class']

# Train classifier
print('\nImplementing Gaussian Naive Bayes Model.')
gnb = GaussianNB()
gnb.fit(
    X_train[feats].values,
    y_train['Class']
)
y_pred = gnb.predict(X_test[feats].values)

print("\nNumber of mislabeled points out of a total {} points : {}, Accuracy: {:05.5f}%"
      .format(
          X_test.shape[0],
          (X_test["Class"] != y_pred).sum(),
          100*(1-(X_test["Class"] != y_pred).sum()/X_test.shape[0])
))

#five fold cross validation
cv = KFold(n_splits=5)
clf = GaussianNB()
X_data=X_data.values
y_data=y_data.values
accuracy=0
for traincv, testcv in cv.split(X_data):
        clf.fit(X_data[traincv], y_data[traincv])
        train_predictions = clf.predict(X_data[testcv])
        acc = accuracy_score(y_data[testcv], train_predictions)
        accuracy+= acc

accuracy = 20*accuracy
print('\n5 Fold Cross Validation Accuracy on Training Set: '+str(accuracy))
