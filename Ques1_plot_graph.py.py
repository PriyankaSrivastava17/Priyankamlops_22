#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


# In[2]:


#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()


# In[3]:


len(digits)


# In[4]:


#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[5]:


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# In[6]:


#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


# In[ ]:


#PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()


# #We are choosing gamma and C as below
# #for 0.001<=gamma<=10
# #for 0.1<=c<=100

# In[ ]:


#PART: setting up hyperparameter
cc=0.3
GAMMA = 0.001
hyper_params = {'gamma':GAMMA,'C':cc}

clf.set_params(**hyper_params)


# In[9]:


#PART: Train model
# Learn the digits on the train subset
clf.fit(X_train, y_train)


# In[10]:


#PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


# In[11]:


#PART: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


# In[12]:


#PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


# In[ ]:




