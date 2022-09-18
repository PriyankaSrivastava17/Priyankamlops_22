#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


# In[2]:



digits = datasets.load_digits()
len(digits.images)


# In[3]:


digits.images.shape


# In[4]:


#PART:Visualization of Original Image
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
for ax, image, label in zip(axes, digits.images, digits.target):
    #ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[5]:


# Visualization of Resize image
_, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 5))
for ax, image, label in zip(axes, digits.images, digits.target):
    image_resized = resize(image, (image.shape[0] // 0.1, image.shape[1] // 0.1),
                       anti_aliasing=True)
    #ax.set_axis_off()
    ax.imshow(image_resized, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[6]:


resi_images = resize(digits.images, (digits.images.shape[0], digits.images.shape[1] //0.1, digits.images.shape[2] //0.1), 
                     anti_aliasing=True)
n_samples = len(resi_images)
data = digits.images.reshape((n_samples, -1))


# In[7]:


resi_images.shape


# In[8]:


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


# In[9]:


clf = svm.SVC()


# In[10]:


#PART: setting up hyperparameter
cc=0.1
GAMMA = 0.001
hyper_params = {'gamma':GAMMA,'C':cc}

clf.set_params(**hyper_params)


# In[11]:


#PART: Train model
# Learn the digits on the train subset
clf.fit(X_train, y_train)


# In[12]:


# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


# In[13]:


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


# In[14]:


#PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


# In[ ]:




