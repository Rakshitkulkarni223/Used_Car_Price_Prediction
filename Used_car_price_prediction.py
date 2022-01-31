#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataframe_train = pd.read_csv('data/dataset.csv')


# In[3]:


train_data = dataframe_train.copy()


# In[4]:


for i in range(0, len(train_data)):
    if train_data['Power'][i] == 'null bhp':
        train_data['Power'][i] = np.nan


# In[5]:


for i in range(0, len(train_data)):
    if train_data['Mileage'][i] == '0.0 kmpl' or train_data['Mileage'][i] == '0.0 km/kg':
        train_data['Mileage'][i] = np.nan


# In[6]:


for i in range(0, len(train_data)):
    if train_data['Engine'][i] == 'null CC' or train_data['Engine'][i] == '0 CC':
        train_data['Engine'][i] = np.nan


# In[7]:


train_data.isnull().sum() 


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.isnull().sum() 


# In[10]:


train_data.info()


# In[11]:


len(train_data)


# In[12]:


train_data.reset_index(inplace=True)


# In[13]:


y = train_data.iloc[:, -1].values


# In[14]:


City = train_data['Location'].unique()


# In[15]:


City


# In[16]:


brand = []
for i in range(0, 5844):
    try:
        k=train_data['Name'][i].split()
        brand.append(k[0].upper())
    except:
        pass


# In[17]:


len(brand)


# In[18]:


Brand = np.array(brand)


# In[19]:


fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(Brand)
ax.set_xlabel("Brands")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')


# In[20]:


Brand = pd.get_dummies(Brand, drop_first=True, dtype=int)


# In[21]:


unique_brands = []
for i in range(0, 5844):
    if brand[i] in unique_brands:
        continue
    else:
        unique_brands.append(brand[i])


# In[22]:


Loc = train_data['Location']

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(Loc)
ax.set_xlabel("Location")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

Loc = pd.get_dummies(Loc, drop_first=True, dtype=int)


# In[23]:


fig = plt.figure(figsize=(7, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(train_data['Seats'])
ax.set_xlabel("Seats")


# In[24]:


fig = plt.figure(figsize=(7, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(train_data['Fuel_Type'])
ax.set_xlabel("Fuel Type")


# In[25]:


fig = plt.figure(figsize=(7, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(train_data['Transmission'])
ax.set_xlabel("Transmission")


# In[26]:


fig = plt.figure(figsize=(7, 7))
fig.add_subplot(1, 1, 1)
ax = sns.countplot(train_data['Owner_Type'])
ax.set_xlabel("Owner Type")


# In[27]:


train_data.replace({'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}, inplace=True)


# In[ ]:





# In[28]:


for i in range(0, 5844):
    try:
        k = train_data['Mileage'][i].split()
        train_data['Mileage'][i] = k[0]
    except:
        print("ok")
        train_data['Mileage'][i]=train_data['Mileage'][i]


# In[29]:


for i in range(0, 5844):
    try:
        k = train_data['Power'][i].split()
        train_data['Power'][i] = k[0]
    except:
        print("ok")
        train_data['Power'][i]=train_data['Power'][i]


# In[30]:


for i in range(0, 5844):
    try:
        k = train_data['Engine'][i].split()
        train_data['Engine'][i] = k[0]
    except:
        print("Ok")
        train_data['Engine'][i]=train_data['Engine'][i]


# In[31]:


train_data['Engine'] = train_data['Engine'].astype(int)
train_data['Power'] = train_data['Power'].astype(float)
train_data['Mileage'] = train_data['Mileage'].astype(float)


# In[32]:


train_data.info()


# In[33]:


Fuel = train_data['Fuel_Type']
Fuel = pd.get_dummies(Fuel, drop_first=True, dtype=int)

Trans = train_data['Transmission']
Trans = pd.get_dummies(Trans, drop_first=True, dtype=int)


# In[34]:


data_train = pd.concat([train_data, Brand, Loc, Fuel, Trans], axis=1)


# In[35]:


data_train.drop(["Name", "Location", "Fuel_Type", 'Transmission', 'Price'], axis=1, inplace=True)


# In[36]:


data_train.drop(['index'], axis=1, inplace=True)


# In[37]:


data_train


# In[38]:


X = data_train.copy()


# In[39]:


plt.figure(figsize=(18, 18))
sns.heatmap(train_data.corr(), annot=True, cmap="RdYlGn")


# In[40]:


from sklearn.ensemble import ExtraTreesRegressor

selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[41]:


plt.figure(figsize=(12, 8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[42]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## RandomForestRegressor

# In[75]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

y_pred_regressor = regressor.predict(X_test)

train_score_regressor=regressor.score(X_train, y_train)
test_score_regressor=regressor.score(X_test, y_test)

print('Train Score: ', train_score_regressor)  
print('Test Score: ', test_score_regressor)


# In[76]:


from sklearn import metrics

mae_regressor= metrics.mean_absolute_error(y_test, y_pred_regressor)
mse_regressor= metrics.mean_squared_error(y_test, y_pred_regressor)
rmse_regressor=np.sqrt(metrics.mean_squared_error(y_test, y_pred_regressor))

print('MAE:',mae_regressor)
print('MSE:', mse_regressor)
print('RMSE:', rmse_regressor)


# In[45]:


sns.distplot(y_test - y_pred_regressor)


# In[46]:


plt.scatter(y_test, y_pred_regressor, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## Linear Regression

# In[77]:


from sklearn.linear_model import LinearRegression  
linear_regressor= LinearRegression()  
linear_regressor.fit(X_train, y_train)  

y_pred_linear = linear_regressor.predict(X_test)

train_linear_regression=linear_regressor.score(X_train, y_train)
test_linear_regression=linear_regressor.score(X_test, y_test)

print('Train Score: ', train_linear_regression)  
print('Test Score: ', test_linear_regression)


# In[78]:


mae_linear_regression=metrics.mean_absolute_error(y_test, y_pred_linear)
mse_linear_regression=metrics.mean_squared_error(y_test, y_pred_linear)
rmse_linear_regression=np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear))

print('MAE:',mae_linear_regression )
print('MSE:', mse_linear_regression)
print('RMSE:',rmse_linear_regression)


# In[51]:


sns.distplot(y_test - y_pred_linear)


# In[52]:


plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## Ridge Regression

# In[79]:


from sklearn.linear_model import Ridge

ridge_regressor = Ridge(alpha=0.01)
ridge_regressor.fit(X_train, y_train) 

y_pred_ridge = ridge_regressor.predict(X_test)

train_ridge_regressor=ridge_regressor.score(X_train, y_train)
test_ridge_regressor=ridge_regressor.score(X_test, y_test)

print('Train Score: ', train_ridge_regressor)  
print('Test Score: ', test_ridge_regressor)


# In[80]:


mae_ridge_regressor= metrics.mean_absolute_error(y_test, y_pred_ridge)
mse_ridge_regressor=metrics.mean_squared_error(y_test, y_pred_ridge)
rmse_ridge_regressor=np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))

print('MAE:',mae_ridge_regressor)
print('MSE:', mse_ridge_regressor)
print('RMSE:', rmse_ridge_regressor)


# In[56]:


sns.distplot(y_test - y_pred_ridge)


# In[57]:


plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## Lasso Regression

# In[81]:


from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train)

y_pred_lasso = model_lasso.predict(X_test)

train_lasso= model_lasso.score(X_train, y_train)
test_lasso=model_lasso.score(X_test, y_test)

print('Train Score: ',train_lasso)  
print('Test Score: ',test_lasso)


# In[82]:


mae_lasso=metrics.mean_absolute_error(y_test, y_pred_lasso)
mse_lasso=metrics.mean_squared_error(y_test, y_pred_lasso)
rmse_lasso=np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso))

print('MAE:', mae_lasso)
print('MSE:', mse_lasso)
print('RMSE:',rmse_lasso)


# In[61]:


sns.distplot(y_test - y_pred_lasso)


# In[62]:


plt.scatter(y_test, y_pred_lasso, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## ElasticNet

# In[83]:


from sklearn.linear_model import ElasticNet

model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 

y_pred_eslasticnet= model_enet.predict(X_test)

train_enet=model_enet.score(X_train, y_train)
test_enet=model_enet.score(X_test, y_test)

print('Train Score: ', train_enet)  
print('Test Score: ', test_enet)


# In[84]:


mae_enet=metrics.mean_absolute_error(y_test, y_pred_eslasticnet)
mse_enet=metrics.mean_squared_error(y_test, y_pred_eslasticnet)
rmse_enet=np.sqrt(metrics.mean_squared_error(y_test, y_pred_eslasticnet))

print('MAE:', mae_enet)
print('MSE:', mse_enet)
print('RMSE:', rmse_enet)


# In[66]:


sns.distplot(y_test - y_pred_eslasticnet)


# In[67]:


plt.scatter(y_test, y_pred_eslasticnet, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## DecisionTreeRegressor

# In[85]:


from sklearn.tree import DecisionTreeRegressor
decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train, y_train)

y_pred_decision_regressor = decision_regressor.predict(X_test)

train_decision_regressor=decision_regressor.score(X_train, y_train)
test_decision_regressor=decision_regressor.score(X_test, y_test)
print('Train Score: ', train_decision_regressor)  
print('Test Score: ',test_decision_regressor)


# In[87]:


mae_decision_regressor=metrics.mean_absolute_error(y_test, y_pred_decision_regressor)
mse_decision_regressor=metrics.mean_squared_error(y_test, y_pred_decision_regressor)
rmse_decision_regressor=np.sqrt(metrics.mean_squared_error(y_test, y_pred_decision_regressor))

print('MAE:', mae_decision_regressor)
print('MSE:', mse_decision_regressor)
print('RMSE:', rmse_decision_regressor)


# In[72]:


sns.distplot(y_test - y_pred_decision_regressor)


# In[73]:


plt.scatter(y_test,y_pred_decision_regressor, alpha=0.5)
plt.title('Decision Tree Regression')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()


# ## Results Visualization

# In[88]:


model_names=["RandomForestRegressor","Linear Regression","Ridge Regression","Lasso Regression",
             "ElasticNet","DecisionTreeRegressor"]

Train_socre=[train_score_regressor,train_linear_regression,train_ridge_regressor,train_lasso,train_enet,train_decision_regressor]

Test_score=[test_score_regressor,test_linear_regression,test_ridge_regressor,test_lasso,test_enet,test_decision_regressor]

mae_all=[mae_regressor,mae_linear_regression,mae_ridge_regressor,mae_lasso,mae_enet,mae_decision_regressor]

mse_all=[mse_regressor,mse_linear_regression,mse_ridge_regressor,mse_lasso,mse_enet,mse_decision_regressor]

rmse_all=[rmse_regressor,rmse_linear_regression,rmse_ridge_regressor,rmse_lasso,rmse_enet,rmse_decision_regressor]


# In[113]:


fig = plt.figure(figsize = (12, 7))
plt.bar(model_names, Train_socre, color ='#46F9A3',
        width = 0.4)

plt.xlabel("Model Names")
plt.ylabel("Train Score")
plt.title("Model Name vs Train Score")
plt.show()


# In[114]:


fig = plt.figure(figsize = (12, 7))
plt.bar(model_names, Test_score, color ='#47F8E9',
        width = 0.4)

plt.xlabel("Model Names")
plt.ylabel("Test Score")
plt.title("Model Name vs Test Score")
plt.show()


# In[115]:


fig = plt.figure(figsize = (12, 7))
plt.bar(model_names, mae_all, color ='#B671FA',
        width = 0.4)

plt.xlabel("Model Names")
plt.ylabel("Mean Absolute Error")
plt.title("Model Name vs Mean Absolute Error")
plt.show()


# In[116]:


fig = plt.figure(figsize = (12, 7))
plt.bar(model_names, mse_all, color ='#F9C953',
        width = 0.4)

plt.xlabel("Model Names")
plt.ylabel("Mean Squared Error")
plt.title("Model Name vs Mean Squared Error")
plt.show()


# In[117]:


fig = plt.figure(figsize = (12, 7))
plt.bar(model_names, rmse_all, color ='#F8446A',
        width = 0.4)

plt.xlabel("Model Names")
plt.ylabel("Root Mean Squared Error")
plt.title("Model Name vs Root Mean Squared Error")
plt.show()

print("Saving the Model....")
import pickle

file = open('car.pkl', 'wb')

pickle.dump(regressor, file)

print("Done!!!")
