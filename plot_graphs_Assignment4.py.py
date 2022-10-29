#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0


# In[16]:


# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

for i in range(5):

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )

    # PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()
    # define the evaluation metric
    metric = metrics.accuracy_score


    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
    )


    # 2. load the best_model
    best_model = load(actual_model_path)

    # PART: Get test set predictions
    # Predict the value of the digit on the test subset
    predicted = best_model.predict(x_test)

    pred_image_viz(x_test, predicted)

    # 4. report the test set accurancy with that best model.
    # PART: Compute evaluation metrics
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )


# In[22]:


#obtained accuracy for five splits/runs
accuracy_svm=[1,1,0.99,0.99,0.98]
import statistics 
print("Standard Deviation of accuracy of svm is % s "% (statistics.stdev(accuracy_svm)))
print("Mean of the accuracy of svm is % s " % (statistics.mean(accuracy_svm))) 


# In[3]:


from sklearn import tree


# In[13]:


digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

for i in range(5):

    x_train1, y_train1, x_dev1, y_dev1, x_test1, y_test1 = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    clf1 = tree.DecisionTreeClassifier()
    clf1 = clf.fit(x_train1, y_train1)
    predicted1=clf1.predict(x_test1)
    print(
        f"Classification report for classifier {clf1}:\n"
        f"{metrics.classification_report(y_test1, predicted1)}\n"
    )


# In[39]:


predicted1=clf1.predict(x_test1)
print('predicted labels:', predicted1)
print('true labels:', y_test1)
import numpy as np

#true_values = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
#predictions = np.array([[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]])

#N = y_test1.shape[1]
N = len(y_test1)
accuracy =(y_test1 == predicted1).sum() / N
TP = ((predicted1 == 1) & (y_test1 == 1)).sum()
FP = ((predicted1 == 1) & (y_test1 == 0)).sum()
precision = TP / (TP+FP)
print('accuracy:', accuracy)
print('true positive:', TP)
print('false positive:', FP)
print('precision:', precision)


# In[ ]:


y_test1


# In[24]:


#obtained accuracy for five splits/runs
accuracy_rf=[0.84,0.86,0.84,0.83,0.84]
import statistics 
print("Standard Deviation of accuracy of rf is % s "% (statistics.stdev(accuracy_rf)))
print("Mean of accuracy of rf is % s " % (statistics.mean(accuracy_rf))) 


# In[ ]:




