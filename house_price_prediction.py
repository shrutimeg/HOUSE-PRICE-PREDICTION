#!/usr/bin/env python
# coding: utf-8

# In[156]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[157]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Shape of train: ", train.shape)
print("Shape of test: ", test.shape)


# In[158]:


train.head(10)


# In[159]:


test.head(10)


# In[160]:


df = pd.concat((train, test))
temp_df = df
print("Shape of df: ", df.shape)


# In[161]:


df.head(6)


# In[162]:


df.tail(6)


# In[163]:


pd.set_option("display.max_columns", 2000)
pd.set_option("display.max_rows", 85)


# In[164]:


df.head(6)


# In[165]:


df.tail(6)


# In[166]:


df.info()


# In[167]:


df.describe()


# In[168]:


df.select_dtypes(include=['int64', 'float64']).columns


# In[169]:


df.select_dtypes(include=['object']).columns


# In[170]:


df = df.set_index("Id")


# In[171]:


plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())


# In[172]:


null_percent = df.isnull().sum()/df.shape[0]*100
null_percent


# In[173]:


train["SalePrice"].describe()


# In[174]:


plt.figure(figsize=(10,8))
bar = sns.distplot(train["SalePrice"])
bar.legend(["Skewness: {:.2f}".format(train['SalePrice'].skew())])


# In[175]:


plt.figure(figsize=(25,25))
ax = sns.heatmap(train.corr(), cmap = "coolwarm", annot=True, linewidth=2)

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[176]:


hig_corr = train.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["SalePrice"]) >= 0.5]
hig_corr_features


# In[177]:


plt.figure(figsize=(10,8))
ax = sns.heatmap(train[hig_corr_features].corr(), cmap = "coolwarm", annot=True, linewidth=3)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[178]:


plt.figure(figsize=(16,9))
for i in range(len(hig_corr_features)):
    if i <= 9:
        plt.subplot(3,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(data=train, x = hig_corr_features[i], y = 'SalePrice')


# In[179]:


missing_col = df.columns[df.isnull().any()]
missing_col


# In[180]:


bsmt_col = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'TotalBsmtSF']
bsmt_feat = df[bsmt_col]
bsmt_feat


# In[181]:


bsmt_feat.info()


# In[182]:


bsmt_feat.isnull().sum()


# In[183]:


bsmt_feat = bsmt_feat[bsmt_feat.isnull().any(axis=1)]
bsmt_feat


# In[184]:


bsmt_feat_all_nan = bsmt_feat[(bsmt_feat.isnull() | bsmt_feat.isin([0])).all(1)]
bsmt_feat_all_nan


# In[185]:


bsmt_feat_all_nan.shape


# In[186]:


qual = list(df.loc[:, df.dtypes == 'object'].columns.values)
qual


# In[187]:


bsmt_feat = bsmt_feat[bsmt_feat.isin([np.nan]).any(axis=1)]
bsmt_feat


# In[188]:


bsmt_feat.shape


# In[189]:


print(df['BsmtFinSF2'].max())
print(df['BsmtFinSF2'].min())


# In[190]:


pd.cut(range(0,1526),5) # create a bucket


# In[191]:


df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]
df_slice


# In[192]:


bsmt_feat.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0] 


# In[193]:


bsmt_feat



# In[194]:


df.update(bsmt_feat)


# In[195]:


bsmt_feat.isnull().sum()


# In[196]:


df.columns[df.isnull().any()]


# In[197]:


garage_col = ['GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',]
garage_feat = df[garage_col]
garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat



# In[198]:


garage_feat.shape


# In[199]:


garage_feat_all_nan = garage_feat[(garage_feat.isnull() | garage_feat.isin([0])).all(1)]
garage_feat_all_nan.shape


# In[200]:


garage_feat_all_nan = garage_feat[(garage_feat.isnull() | garage_feat.isin([0])).all(1)]
garage_feat_all_nan.shape


# In[201]:


garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat


# In[202]:


garage_feat.isnull().any()


# In[203]:


df.update(garage_feat)


# In[204]:


df.columns[df.isnull().any()]


# In[205]:


df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[206]:


df.columns[df.isnull().any()]


# In[207]:


df[df['MasVnrArea'].isnull() == True]['MasVnrType'].unique()


# In[208]:


df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull() == True), 'MasVnrArea'] = 0


# In[209]:


df.isnull().sum()/df.shape[0] * 100


# In[210]:


lotconfig = ['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3']
for i in lotconfig:
    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i) , df[df['LotConfig'] == i] ['LotFrontage'].mean(), df['LotFrontage'])


# In[211]:


df.isnull().sum()


# In[212]:


df.columns


# In[213]:


feat_dtype_convert = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
for i in feat_dtype_convert:
    df[i] = df[i].astype(str)


# In[214]:


df['MoSold'].unique()


# In[215]:


import calendar
df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])


# In[216]:


df['MoSold'].unique()


# In[217]:


quan = list(df.loc[:, df.dtypes != 'object'].columns.values)


# In[218]:


quan


# In[219]:


len(quan)


# In[220]:


obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
obj_feat


# In[221]:


from pandas.api.types import CategoricalDtype
df['BsmtCond'] = df['BsmtCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes


# In[222]:


df['BsmtCond'].unique()


# In[223]:


df['BsmtExposure'] = df['BsmtExposure'].astype(CategoricalDtype(categories=['NA', 'Mn', 'Av', 'Gd'], ordered = True)).cat.codes


# In[224]:



df['BsmtExposure'].unique()


# In[225]:


df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterQual'] = df['ExterQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod','Min2','Min1', 'Typ'], ordered = True)).cat.codes
df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N', 'P', 'Y'], ordered = True)).cat.codes
df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO', 'NASeWa', 'NASeWr', 'AllPub'], ordered = True)).cat.codes


# In[226]:


df['Utilities'].unique()


# In[227]:


skewed_features = ['1stFlrSF',
 '2ndFlrSF',
 '3SsnPorch',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtFullBath',
 'BsmtHalfBath',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Fireplaces',
 'FullBath',
 'GarageArea',
 'GarageCars',
 'GrLivArea',
 'HalfBath',
 'KitchenAbvGr',
 'LotArea',
 'LotFrontage',
 'LowQualFinSF',
 'MasVnrArea',
 'MiscVal',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']


# In[228]:


quan == skewed_features


# In[229]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[230]:


df_back = df


# In[231]:


for i in skewed_features:
    df[i] = np.log(df[i] + 1)
    


# In[232]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[233]:


SalePrice = np.log(train['SalePrice'] + 1)


# In[234]:


obj_feat = list(df.loc[:,df.dtypes == 'object'].columns.values)
len(obj_feat)


# In[235]:


dummy_drop = []
clean_df = df
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[236]:


df.shape


# In[237]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # model building
# 

# In[238]:


train_len = len(train)


# In[239]:


X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print(X_train.shape)
print(X_test.shape)
print(len(y_train))


# In[240]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# 
# 
# # Linear regression

# In[241]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[242]:


cross_validation = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
print("Cross validation accuracy of LR model = ", cross_validation)
print("\nCross validation mean accuracy of LR model = ", cross_validation.mean())


# In[243]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[244]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# In[245]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)


# # DECISION TREE

# In[246]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)
           
           


# In[257]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)


# # Bagging & Boosting¶
# 

# In[259]:


from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)


# In[261]:


test_model(br_reg)


# In[262]:


get_ipython().system('pip install xgboost')


# In[263]:


import xgboost
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)


# In[264]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[269]:


y_pred



# In[270]:


submit_test1 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test1.columns=['Id', 'SalePrice']


# In[271]:


submit_test1


# In[272]:


submit_test1.to_csv('sample_submission.csv', index=False )


# In[274]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.cv


# In[ ]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# In[275]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[ ]:


y_pred


# In[ ]:


submit_test3 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test3.columns=['Id', 'SalePrice']


# In[277]:


submit_test3.to_csv('sample_submission.csv', index=False)
submit_test3


# # XGBoost Parameter Tuning¶
# 

# In[278]:


xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,
 mon_child_weight= 2,
 max_depth= 4,
 learning_rate= 0.05,
 booster= 'gbtree')

test_model(xgb2_reg)


# In[279]:


xgb2_reg.fit(X_train,y_train)
y_pred_xgb_rs=xgb2_reg.predict(X_test)


# In[280]:


np.exp(y_pred_xgb_rs).round(2)


# In[282]:


y_pred_xgb_rs = np.exp(xgb2_reg.predict(X_test)).round(2)
xgb_rs_solution = pd.concat([test['Id'], pd.DataFrame(y_pred_xgb_rs)], axis=1)
xgb_rs_solution.columns=['Id', 'SalePrice']
xgb_rs_solution.to_csv('sample_submission.csv', index=False)


# In[283]:


xgb_rs_solution


# # Selection To Improve Accuracy

# In[284]:


plt.figure(figsize=(9,16))
corr_feat_series = pd.Series.sort_values(train.corrwith(train.SalePrice))
sns.barplot(x=corr_feat_series, y=corr_feat_series.index, orient='h')


# In[285]:


df_back1 = df_back


# In[286]:


df_back1.to_csv('df_for_feature_engineering.csv', index=False)


# In[287]:


list(corr_feat_series.index)


# # Feature Selection

# In[288]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[289]:


df = pd.read_csv('df_for_feature_engineering.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df


# # Drop Feature 
# 

# In[290]:


df = df.drop(['YrSold',
 'LowQualFinSF',
 'MiscVal',
 'BsmtHalfBath',
 'BsmtFinSF2',
 '3SsnPorch',
 'MoSold'],axis=1)


# In[291]:


quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
quan


# In[292]:


skewd_feat = ['1stFlrSF',
 '2ndFlrSF',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFullBath',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Fireplaces',
 'FullBath',
 'GarageArea',
 'GarageCars',
 'GrLivArea',
 'HalfBath',
 'KitchenAbvGr',
 'LotArea',
 'LotFrontage',
 'MasVnrArea',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']


# In[293]:


for i in skewd_feat:
    df[i] = np.log(df[i] + 1)
    
SalePrice = np.log(train['SalePrice'] + 1)


# In[294]:


df


# In[295]:


obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
print(len(obj_feat))

obj_feat


# In[296]:


dummy_drop = []
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[297]:


df.shape


# In[298]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # Model Bulding
# 

# In[299]:


train_len = len(train)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print("Shape of X_train: ", len(X_train))
print("Shape of X_test: ", len(X_test))
print("Shape of y_train: ", len(y_train))


# In[300]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# # linear model

# In[301]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[302]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[303]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# In[304]:


from sklearn.svm import SVR
svr = SVR(kernel='rbf')
test_model(svr)


# # Svm Hyper Parameter Tuning
# 
# 

# In[305]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_


# In[306]:


rand_search.best_estimator_


# In[308]:


svr_reg1=SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg1)


# In[309]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# # XGBoost
# 
# 

# In[310]:


import xgboost
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)


# In[311]:


xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,
 mon_child_weight= 2,
 max_depth= 4,
 learning_rate= 0.05,
 booster= 'gbtree')

test_model(xgb2_reg)


# In[329]:


xgb2_reg.fit(X_train,y_train)
y_pred = np.exp(xgb2_reg.predict(X_test)).round(2)
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test


# In[327]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test


# # Model Save
# 

# In[324]:


import pickle
      
pickle.dump(svr_reg, open('model_house_price_prediction.csv', 'wb'))
model_house_price_prediction = pickle.load(open('model_house_price_prediction.csv', 'rb'))
model_house_price_prediction.predict(X_test)


# In[323]:


test_model(model_house_price_prediction.csv)



# In[ ]:




