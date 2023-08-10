#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# TASK 2: Build a model to detect fraudulent credit card transactions. Use a
# dataset containing information about credit card transactions, and
# experiment with algorithms like Logistic Regression, Decision Trees,
# or Random Forests to classify transactions as fraudulent or
# legitimate.


# In[1]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from warnings import filterwarnings
filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None    
pd.options.display.float_format = '{:.6f}'.format
from sklearn.model_selection import train_test_split
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from scipy import stats
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge


# In[2]:


data1 = pd.read_csv('fraudTrain.csv')
data1.head()


# In[3]:


data2 = pd.read_csv('fraudTest.csv')
data2.head()


# In[4]:


combined_data = pd.concat([data1,data2], axis = 0)
combined_data.head()


# In[5]:


combined_data.info()


# In[7]:


combined_data.reset_index(inplace = True)
combined_data.head()


# In[8]:


combined_data = combined_data.drop(['index', 'Unnamed: 0'], axis = 1)


# In[9]:


combined_data.describe()


# In[10]:


plt.figure(figsize = (15,5), dpi = 100)
sns.countplot(x = combined_data['is_fraud'])


# In[11]:


plt.figure(figsize = (15,5), dpi = 100)
sns.countplot(x = 'gender', hue = 'is_fraud', data = combined_data)


# In[13]:


plt.figure(figsize = (20,6), dpi = 200)
sns.countplot(x = 'category', hue = 'is_fraud', data = combined_data)
plt.xticks(rotation = 90)
plt.show()


# In[14]:


feature = combined_data.drop(['is_fraud'], axis = 1)
target = combined_data['is_fraud']


# In[15]:


from sklearn.preprocessing import OrdinalEncoder
columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
        'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
encoder = OrdinalEncoder()
feature[columns] = encoder.fit_transform(feature[columns])


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature = scaler.fit_transform(feature)


# In[17]:


target = combined_data[['is_fraud']].values


# In[18]:


print('Independent Features: ', feature.shape)
print('Dependent Features: ',target.shape)


# In[19]:


combined_data['is_fraud'].value_counts()


# In[20]:


from imblearn.under_sampling import NearMiss
nm_sampler = NearMiss()
feature_sampled, target_sampled = nm_sampler.fit_resample(feature, target)

print('Data: ', feature_sampled.shape)
print('Labels: ', target_sampled.shape)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(feature_sampled, target_sampled, 
                                                    random_state=2, test_size = 0.2)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[ ]:


# Logistic Regression


# In[34]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[32]:


from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)


# In[35]:


pred_train = lr_classifier.predict(X_train)
pred_test  = lr_classifier.predict(X_test)
print('Training Accuracy : ', accuracy_score(y_train, pred_train))
print('Testing  Accuracy : ', accuracy_score(y_test, pred_test))


# In[36]:


print('Training Set f1 score : ', f1_score(y_train, pred_train))
print('Testing  Set f1 score : ', f1_score(y_test, pred_test))
print()
print('Test set precision : ', precision_score(y_test, pred_test))
print('Test set recall    : ', recall_score(y_test, pred_test))


# In[37]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion_matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[38]:


cm = confusion_matrix(y_test, pred_test)
plt.figure(figsize = (10,5), dpi = 100)
sns.set(rc = {'axes.grid' : False})
plot_confusion_matrix(cm, classes = ['non_fraudulent(0)','fraudulent(1)'])


# In[ ]:


# Decision Tree Classification


# In[40]:


from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(max_depth = 50, random_state = 100)
dt_classifier.fit(X_train, y_train)


# In[41]:


pred_train = dt_classifier.predict(X_train)
pred_test  = dt_classifier.predict(X_test)
print('Training Accuracy : ', accuracy_score(y_train, pred_train))
print('Testing  Accuracy : ', accuracy_score(y_test, pred_test))


# In[42]:


print('Training Set f1 score : ', f1_score(y_train, pred_train))
print('Testing  Set f1 score : ', f1_score(y_test, pred_test))
print()
print('Test set precision : ', precision_score(y_test, pred_test))
print('Test set recall    : ', recall_score(y_test, pred_test))


# In[43]:


cm = confusion_matrix(y_test, pred_test)
plt.figure(figsize = (10,5), dpi = 100)
sns.set(rc = {'axes.grid' : False})
plot_confusion_matrix(cm, classes = ['non_fraudulent(0)','fraudulent(1)'])


# In[ ]:


# Random Forest Classification


# In[45]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth = 12, random_state = 2)
rf_classifier.fit(X_train, y_train)


# In[46]:


pred_train = rf_classifier.predict(X_train)
pred_test  = rf_classifier.predict(X_test)
print('Training Set Accuracy : ', accuracy_score(y_train, pred_train))
print('Testing Set Accuracy  : ', accuracy_score(y_test, pred_test))


# In[47]:


cm = confusion_matrix(y_test, pred_test)
plt.figure(figsize = (10,6), dpi = 100)
sns.set(rc = {'axes.grid' : False})
plot_confusion_matrix(cm, classes = ['non_fraudulent(0)','fraudulent(1)'])

