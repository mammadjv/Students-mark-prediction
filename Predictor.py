import numpy as np
import pandas as pd
import sklearn
from sklearn import tree, preprocessing, ensemble
import sys
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.sgd_fast import Regression
import matplotlib.pyplot as plt
from sklearn.svm.classes import SVC
from matplotlib import gridspec
from sklearn.linear_model.coordinate_descent import Lasso
import math as math
import random as random
from random import randrange


def encode_field(train,test,scale):

    dfTrain = pd.DataFrame(train)
    dfTest = pd.DataFrame(test)
    dfTemp = pd.DataFrame(train).append(dfTest)

    dftrainHeaderList = list(dfTemp)

    for header in dftrainHeaderList:
            encoder = preprocessing.LabelEncoder()
            encoder.fit(dfTemp[header].values)
            dfTrain[header] = encoder.transform(dfTrain[header].values)
            dfTest[header] = encoder.transform(dfTest[header].values)

    return train , test

def featureSelector(data,trainHeaderList,target,selectorType):
    dataFrame = pd.DataFrame(data)
    if(selectorType == 'VT'):
        cols = dataFrame.columns
        pi = 0.6
        selector = VarianceThreshold(threshold=(pi*(1-pi)))
        values = selector.fit_transform(dataFrame)
        labels = list()
        i = 0
        for x in selector.get_support(indices=False):
            if x:
                labels.append(trainHeaderList.__getitem__(i))
            i += 1
        return pd.DataFrame(values , columns=labels)

    elif(selectorType == 'KB'):
        selector = SelectKBest(chi2, k=6)
        values = selector.fit_transform(dataFrame, target)
        labels = list()
        i = 0
        for x in selector.get_support(indices=False):
            if x:
                labels.append(trainHeaderList.__getitem__(i))
            i += 1
        return pd.DataFrame(values, columns=labels)

    elif(selectorType == 'SVC'):
        svc = SVC(kernel="linear", C=1)
        selector = RFE(estimator=svc, n_features_to_select=20, step=0.5, verbose=5)
        values =selector.fit_transform(dataFrame, target)
        labels = list()
        i = 0
        for x in selector.get_support(indices=False):
            if x:
                labels.append(trainHeaderList.__getitem__(i))
            i += 1
        return pd.DataFrame(values, columns=labels)

def K_fold_CrossValidation(k , dataFrame , target , regressorType):
    trainDataSet = pd.DataFrame(dataFrame)
    regressor = Regression
    if(regressorType == "GDB"):
        regressor = ensemble.GradientBoostingRegressor(n_estimators=1000, max_depth=4, min_samples_split=2,
                                            learning_rate=0.001, loss='ls')
    if(regressorType == "LN"):
        regressor = LinearRegression()
    if (regressorType == "SVR"):
        regressor = SVR(kernel='linear', C=1e3)
    if (regressorType == "LS"):
        regressor = Lasso(alpha=0.001, normalize=True)

    part_size = int(np.floor(len(trainDataSet) / float(k)))
    best_part = 0
    min_error = 1000

    for i in range(0,k):
        trainSubSet = trainDataSet[:][0:i*part_size].append(trainDataSet[:][(i+1)*part_size:])
        testSubSet = trainDataSet[i*part_size:(i+1)*part_size]
        targetSubSet = target[:][0:i*part_size].append(target[:][(i+1)*part_size:])
        desireValue = target[i*part_size:(i+1)*part_size]

        regressor.fit(trainSubSet,targetSubSet.values.ravel())
        predictedValue = regressor.predict(testSubSet)

        value = 0.00
        for i in range(len(predictedValue)):
             print predictedValue[i]
             print desireValue.values[i]
             value += ((predictedValue[i] - desireValue.values[i]) ** 2)
             print "value  -- " , value
        error  = math.sqrt(value / part_size)

        print "error = " , error
        if(error < min_error):
            min_error = error
            best_part = i

    print("min_error =   " , min_error )
    trainSubSet = trainDataSet[:][0:best_part*part_size].append(trainDataSet[:][(best_part+1)*part_size:])
    targetSubSet = target[:][0:best_part*part_size].append(target[:][(best_part+1)*part_size:])
    regressor.fit(trainSubSet,targetSubSet.values.ravel())
    return regressor

def fillNans(data):
    train = pd.DataFrame(data)
    trainHeaderList = list(train)
    for header in trainHeaderList:
        class_mean = train[header][train[header] != 0.0].mean()
        class_zero = np.size(train[header][train[header] == 0.0])
        class_coef = 1 - (float(class_zero) / float(len(train[header])))
        for i in xrange(1, len(train)):
            if (train[header].iloc[i] < 2):
                student_mark_mean = train[:].iloc[i].mean()
                student_zero = 0
                student_mark_coef = 0.5
                for rowvalues in train[:].iloc[i]:
                    if (rowvalues == 0):
                        student_zero += 1
                if (student_zero >= 3):
                    student_mark_coef = 0.00
                else:
                    student_mark_coef = float(6 - student_zero) / 6.00
                if (student_mark_coef == 0.00):
                    train[header].iloc[i] = class_mean
                elif (class_coef < 0.2):
                    train[header].iloc[i] = max(student_mark_mean, 9.5)
                else:
                    train[header].iloc[i] = max(
                        (student_mark_coef / (student_mark_coef + class_coef)) * student_mark_mean + (
                        class_coef / (student_mark_coef + class_coef)) * class_mean, 10)
    return train

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


trainHeaderList = list(train)
testHeaderList = list(test)

target = pd.DataFrame(train[['G7']])
train = pd.DataFrame(train.drop('G7',axis=1))

# del train['G3']

# trainHeaderList.remove('G3')
trainHeaderList.remove('G7')

train = fillNans(train)
test = fillNans(test)



# sys.exit()

#i = 321
#plt.figure(figsize=(12, 9))
#for header in trainHeaderList:
#    ax = plt.subplot(i)
#    ax.set_title(header)
#    ax.stem(train[header],target, '.')
#    i += 1

#plt.subplots_adjust(hspace=.5)
#plt.show()

trainHeaderList = list(train)
testHeaderList = list(test)

selectedFeatures = featureSelector(train,trainHeaderList,target,'VT')

selectedFeaturesHeader = list(selectedFeatures)


for header in testHeaderList:
    if(header in selectedFeaturesHeader):
        continue
    else:
        del test[header]


regressor = K_fold_CrossValidation(10,selectedFeatures,target,"SVR")

prediction_value = regressor.predict(test)

testId = np.arange(1,np.size(prediction_value)+1,1).astype(int)

my_solution = pd.DataFrame(prediction_value, testId, columns=["G7"])

my_solution.to_csv("P1_submission.csv", index_label=["Id"])

plt.plot(prediction_value , '.')
# plt.show()
plt.plot(target)
plt.show()