#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import eli5
from eli5.sklearn import PermutationImportance
import os
import re
import warnings
warnings.filterwarnings("always")

from sklearn.linear_model import Lasso

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

pd.options.display.max_columns= 999
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# In[3]:


os.chdir("/data")


# In[4]:


kcps = pd.read_csv("KCPS_r2_0.01.csv")
kcps = kcps.drop(["IID"],axis=1)


# In[73]:


train = pd.read_table("phenotype.txt")
train = train.drop(["IID","SCOLON","SRECTM","SPROST","STHROI","SBREAC","SLUNG","SSTOMA","SLIVER","SCRC","FVC","FEV1","COLON","LIVER","LUNG","PROST","THROI","BREAC","RECTM","CRC","GOT_B","GPT_B","GGT_B","URIC_B","BIL","WBC","CREAT","TG_B","FBS_B"],axis=1)
train = train.drop(["PCAN80","PCAN81","PCAN82","PCAN83","PCAN84","PCAN86","PCAN89","MDM_B","MHTN_B","MLPD_B","PHTN_B","PDM_B","PLPD_B","FCAN80","FCAN81","FCAN82","FCAN83","FCAN84","FCAN86","FCAN89"],axis=1)


# In[1389]:


# 두 변수를 추가할 때 사용
null_columns = ["PCAN81","FCAN81"]
for columns in null_columns:
    train[columns] = train[columns].fillna(0)
    train[columns] = train[columns].astype("int")
#null_columns = ["PCAN80","PCAN81","PCAN82","PCAN83","PCAN84","PCAN86","PCAN89","MDM_B","MHTN_B","MLPD_B","PHTN_B","PDM_B","PLPD_B","FCAN80","FCAN81","FCAN82","FCAN83","FCAN84","FCAN86","FCAN89"]
#for columns in null_columns:
 #   train[columns] = train[columns].fillna(0)


# In[74]:


# Phenotype 데이터 사용
train = train.loc[(train["SMOK_B"].notnull()) |(train["SMOKA_MOD_B"].notnull()),]
train = train.loc[(train["ALCO_B"].notnull()) |(train["ALCO_AMOUNT_B"].notnull()),]
train.loc[(train["SMOK_B"]==1) & (train["SMOKA_MOD_B"].isna()),"SMOKA_MOD_B"] = 0.7706
train.loc[((train["SMOK_B"]==2) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==1),"SMOKA_MOD_B" ] = 11.99
train.loc[((train["SMOK_B"]==2) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==2),"SMOKA_MOD_B" ] = 4.521
train.loc[((train["SMOK_B"]==3) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==1),"SMOKA_MOD_B" ] = 14.377
train.loc[((train["SMOK_B"]==3) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==2),"SMOKA_MOD_B" ] = 7.7227
train.loc[((train["ALCO_B"]==1) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==1)),"ALCO_AMOUNT_B"] = 3.7678
train.loc[((train["ALCO_B"]==1) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==2)),"ALCO_AMOUNT_B"] = 1.1029
train.loc[((train["ALCO_B"]==2) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==1)),"ALCO_AMOUNT_B"] = 24.1013
train.loc[((train["ALCO_B"]==2) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==2)),"ALCO_AMOUNT_B"] = 7.4422
train.loc[(train["ALCO_B"].isna()) & (train["ALCO_AMOUNT_B"]==0),"ALCO_B"] = 1
train["BMI"] = train["WT_B"] / (train["HT_B"]**2)
train["PCAN81_FCAN81"] = train["PCAN81"].astype(str) + "_" + train["FCAN81"].astype(str)
train = train.dropna()
train,test = train_test_split(train,test_size=0.2,random_state=156,shuffle=True,stratify=train["STOMA"])


# In[57]:


# SNPs 데이터 사용
train = pd.read_table("phenotype.txt")
train = train[["FID","STOMA"]]
train = train.merge(kcps,on="FID",how="left")
train,test = train_test_split(train,test_size=0.2,random_state=156,shuffle=True,stratify=train["STOMA"])


# In[66]:


# Phenotype + SNPs 사용
train = train.loc[(train["SMOK_B"].notnull()) |(train["SMOKA_MOD_B"].notnull()),]
train = train.loc[(train["ALCO_B"].notnull()) |(train["ALCO_AMOUNT_B"].notnull()),]
train.loc[(train["SMOK_B"]==1) & (train["SMOKA_MOD_B"].isna()),"SMOKA_MOD_B"] = 0.7706
train.loc[((train["SMOK_B"]==2) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==1),"SMOKA_MOD_B" ] = 11.99
train.loc[((train["SMOK_B"]==2) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==2),"SMOKA_MOD_B" ] = 4.521
train.loc[((train["SMOK_B"]==3) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==1),"SMOKA_MOD_B" ] = 14.377
train.loc[((train["SMOK_B"]==3) & (train["SMOKA_MOD_B"].isna())) & (train["SEX1"]==2),"SMOKA_MOD_B" ] = 7.7227
train.loc[((train["ALCO_B"]==1) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==1)),"ALCO_AMOUNT_B"] = 3.7678
train.loc[((train["ALCO_B"]==1) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==2)),"ALCO_AMOUNT_B"] = 1.1029
train.loc[((train["ALCO_B"]==2) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==1)),"ALCO_AMOUNT_B"] = 24.1013
train.loc[((train["ALCO_B"]==2) &(train["ALCO_AMOUNT_B"].isna()) & (train["SEX1"]==2)),"ALCO_AMOUNT_B"] = 7.4422
train.loc[(train["ALCO_B"].isna()) & (train["ALCO_AMOUNT_B"]==0),"ALCO_B"] = 1
train["BMI"] = train["WT_B"] / (train["HT_B"]**2)
train["PCAN81_FCAN81"] = train["PCAN81"].astype(str) + "_" + train["FCAN81"].astype(str)
train = train.dropna()
train = train.merge(kcps,on="FID",how="left")
train,test = train_test_split(train,test_size=0.2,random_state=156,shuffle=True,stratify=train["STOMA"])


# In[75]:


train2 = train.drop(["FID","STOMA"],axis=1)
test2 = test.drop(["FID","STOMA"],axis=1)

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

train2.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train2.columns.values]
test2.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test2.columns.values]

train2 = train2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
test2 = test2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# In[1392]:


# 두 변수를 추가할 때 사용
lb = LabelEncoder()
train2["PCAN81_FCAN81"] = lb.fit_transform(train2["PCAN81_FCAN81"])
test2["PCAN81_FCAN81"] = lb.transform(test2["PCAN81_FCAN81"])


# In[1386]:


union_columns = set(train2.columns).union(set(inter_columns))


# In[1387]:


len(union_columns)


# In[1355]:


# SNPs 데이터에서 381개로 줄인 변수 사용
train2 = train2[inter_columns]
test2 = test2[inter_columns]


# In[1393]:


# Phenotype + SNPs 데이터에서 381개로 줄인 변수 사용
train2 = train2[union_columns]
test2 = test2[union_columns]


# In[77]:


def objective(trial:Trial) -> float:
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    n_estimators =  trial.suggest_int('n_estimators', 100, 500)
   
    rf = RandomForestClassifier(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,n_jobs=2,random_state=25)
    X_train,X_test,y_train,y_test = train_test_split(train2,train["STOMA"],test_size=0.2,random_state=156,stratify=train["STOMA"])

    
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_test)[:,1]
    rf_score = roc_auc_score(y_test,rf_pred)
    return rf_score

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="rf_parameter_opt",
    direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)


# In[61]:


rf_params = {'max_depth': 4, 'max_leaf_nodes': 951, 'n_estimators': 393}


# In[70]:


rf_params = {'max_depth': 2, 'max_leaf_nodes': 305, 'n_estimators': 310}


# In[78]:


rf_params = {'max_depth': 8, 'max_leaf_nodes': 22, 'n_estimators': 488}


# In[79]:


sf = StratifiedKFold(n_splits=5,shuffle=True,random_state=156)
scores = []
importances_rf = 0
i=1
for train_idx,valid_idx in sf.split(train2,train["STOMA"]):
    X_train,X_valid = train2.iloc[train_idx] , train2.iloc[valid_idx]
    y_train,y_valid = train["STOMA"].iloc[train_idx], train["STOMA"].iloc[valid_idx]

    rf = RandomForestClassifier(**rf_params,random_state=156,n_jobs=-1)
    rf.fit(X_train,y_train)
    importances_rf += rf.feature_importances_ / 5
    
    pred = rf.predict_proba(X_valid)
    score = roc_auc_score(y_valid,pred[:,1])
    scores.append(score)
    print("{}번째 폴드 점수 : {}".format(i,score))
    i+=1
print("검증 평균 점수 : {}".format(np.mean(scores)))
print("테스트 점수 : {}".format(roc_auc_score(test["STOMA"],rf.predict_proba(test2)[:,1])))


# In[72]:


sf = StratifiedKFold(n_splits=5,shuffle=True,random_state=156)
pred = 0
score = 0
importances_lgbm = 0
i=1
for train_idx,valid_idx in sf.split(train2,train["STOMA"]):
    X_train,X_valid = train2.iloc[train_idx] , train2.iloc[valid_idx]
    y_train,y_valid = train["STOMA"].iloc[train_idx], train["STOMA"].iloc[valid_idx]

    lgbm = LGBMClassifier()
    lgbm.fit(X_train,y_train,eval_set=(X_valid,y_valid),early_stopping_rounds=30,eval_metric="auc",verbose=100)
    importances_lgbm += lgbm.feature_importances_ / 5
    pred  += lgbm.predict_proba(test2) / 5
    score += lgbm.best_score_["valid_0"]["auc"] / 5

print("검증 점수 : {}".format(score))
print("테스트 점수 : {}".format(roc_auc_score(test["STOMA"],pred[:,1])))


# In[64]:


sf = StratifiedKFold(n_splits=5,shuffle=True,random_state=156)
pred = 0
score = 0
i=1
importances_xgb = 0
for train_idx,valid_idx in sf.split(train2,train["STOMA"]):
    X_train,X_valid = train2.iloc[train_idx] , train2.iloc[valid_idx]
    y_train,y_valid = train["STOMA"].iloc[train_idx], train["STOMA"].iloc[valid_idx]

    xgb = XGBClassifier()
    xgb.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],early_stopping_rounds=30,eval_metric="auc",verbose=100)
    importances_xgb += xgb.feature_importances_ / 5
    pred  += xgb.predict_proba(test2) / 5
    score += xgb.best_score / 5

print("검증 점수 : {}".format(score))
print("테스트 점수 : {}".format(roc_auc_score(test["STOMA"],pred[:,1])))


# In[1191]:


rf_importance = pd.DataFrame({"columns":train2.columns,"importance":importances_rf}).sort_values("importance",ascending=False)
rf_columns = rf_importance[rf_importance["importance"]>0]


# In[1192]:


lgbm_importance = pd.DataFrame({"columns":train2.columns,"importance":importances_lgbm}).sort_values("importance",ascending=False)
lgbm_columns = lgbm_importance[lgbm_importance["importance"]>0]


# In[1193]:


xgb_importance = pd.DataFrame({"columns":train2.columns,"importance":importances_xgb}).sort_values("importance",ascending=False)
xgb_columns = xgb_importance[xgb_importance["importance"]>0]


# In[1194]:


inter_columns = (set(rf_columns["columns"].values).intersection(lgbm_columns["columns"].values)).intersection(set(xgb_columns["columns"].values))
len(inter_columns)


# In[ ]:





# In[494]:





# In[495]:





# In[496]:





# In[497]:





# In[498]:





# In[ ]:




