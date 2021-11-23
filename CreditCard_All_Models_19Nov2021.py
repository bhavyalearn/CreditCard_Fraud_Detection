#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier



from sklearn.metrics import confusion_matrix,classification_report,make_scorer
from sklearn.metrics import accuracy_score,recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, ShuffleSplit
from sklearn.pipeline import Pipeline


# ### Read data

# In[50]:


credit_card_data = pd.read_csv(r'F:\ML\Data\Credit card data\creditcard.csv')
credit_card_data.shape


# In[8]:


credit_card_data.head()


# In[9]:


credit_card_data.info()


# In[10]:


credit_card_data.isnull().sum()


# In[11]:


credit_card_data['Class'].value_counts()


# In[13]:


correlation = credit_card_data.corr()
plt.figure(figsize=(17,12))
sns.heatmap(correlation,annot=True)
plt.show()


# In[96]:


high_correlation = correlation[ ((correlation > 0.3) | (correlation < -0.3)) & (correlation != 1)]
plt.figure(figsize=(17,12))
sns.heatmap(high_correlation,annot=True)
plt.show()


# In[93]:


hr = high_correlation.drop(['Time'],axis=0)
hr = hr.drop(['Time'],axis=1)
plt.figure(figsize=(17,12))
sns.heatmap(hr,annot=True)
plt.show()


# ### add more data visualization here

# ### Model building

# In[46]:


# create Model's instances

lr = LogisticRegression(solver='lbfgs')
randomForest = RandomForestClassifier()
gBooster = GradientBoostingClassifier() 
adaBoost = AdaBoostClassifier()
bagging = BaggingClassifier()

listOfModels = [lr,randomForest,gBooster, adaBoost, bagging]
modelNames = ['Logistic Regression','Random Forest', 'Gradient Booster', 'Ada Boost', 'Bagging']

n_folds  = 5
shuffle_split = ShuffleSplit(n_splits = n_folds,test_size= 0.30, train_size = 0.60)
std_scaler = StandardScaler()


# In[37]:


x = credit_card_data.drop(['Class'],axis=1)
y = credit_card_data['Class']


# In[47]:


mla_columns = ['MLA Name','MLA Parameters','MLA Train Accuracy Mean','MLA Test Accuracy Mean', 'MLA Time']
mla_comparision = pd.DataFrame(columns = mla_columns)

mla_predict = credit_card_data.index.values
#train_size = x.shape[0]
number_of_models = len(listOfModels)

#oof_predictions = np.zeors((train_size,no_of_models))

#scores = []

row_index = 0

for n, model in enumerate(listOfModels):
    model_pipeline = Pipeline(steps = [('Scaler',std_scaler),
                                       ('Estimator',model)])
    mla_name = model.__class__.__name__
    mla_comparision.loc[row_index,'MLA Name'] = mla_name
    
    mla_comparision.loc[row_index,'MLA Parameters'] = str(model.get_params())
    
    cv_results = cross_validate(model,x,y,cv=shuffle_split,return_train_score = True)
    
    mla_comparision.loc[row_index,'MLA Time'] = cv_results['fit_time'].mean()
    mla_comparision.loc[row_index,'MLA Train Accuracy Mean']= round(cv_results['train_score'].mean(),2)
    mla_comparision.loc[row_index,'MLA Test Accuracy Mean'] = round(cv_results['test_score'].mean(),2)
    
    model_pipeline.fit(x,y)
    
    mla_predict = model_pipeline.predict(x)
    row_index += 1


# In[49]:



mla_comparision.sort_values(by=['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(mla_comparision)
    


# ### Understanding Feature Importance

# In[83]:


feature_names = x.columns
modelNames_for_fi = ['Random Forest', 'Gradient Booster', 'Ada Boost']
listOfModels_for_fi = [randomForest,gBooster, adaBoost]

feature_importance = pd.DataFrame(columns = modelNames_for_fi, index = feature_names)

for n,model in enumerate(listOfModels_for_fi):
    feature_importance[modelNames_for_fi[n]] = listOfModels_for_fi[n].feature_importances_


# In[84]:


print(feature_importance)

