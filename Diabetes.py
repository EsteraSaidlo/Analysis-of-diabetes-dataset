# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:49:02 2020

@author: Esterka
"""

import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
    %matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import squarify
from sklearn import linear_model



cukrzyca = pd.read_csv("diabetes.csv")

cukrzyca.head()
cukrzyca.info()
desc =cukrzyca.describe()

sum(cukrzyca.iloc[:,2]==0)
sum(cukrzyca.iloc[:,3]==0)
sum((cukrzyca.iloc[:,2]==0)&(cukrzyca.iloc[:,4]==0))
sum(cukrzyca.iloc[:,5]==0)
sum(cukrzyca.iloc[:,6]==0)
sum(cukrzyca.iloc[:,7]==0)


        
cukrzyca[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=cukrzyca[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)
print(cukrzyca.isna().sum())

pip install quilt
quilt install ResidentMario/missingno_data

cukrzyca.groupby("Outcome").size()
cukrzyca.hist(figsize=(10,8))
cukrzyca.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))

corr = cukrzyca[cukrzyca.columns].corr()
sns.heatmap(corr, annot = True )

missing_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

for feature in missing_columns:
    cukrzyca[feature + '_imp'] = cukrzyca[feature]
    cukrzyca = random_imputation(cukrzyca, feature)

deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

for feature in missing_columns:
    deter_data["Det" + feature] = cukrzyca[feature + "_imp"]
    parameters = list(set(cukrzyca.columns) - set(missing_columns) - {feature + '_imp'})
    
    #Create a Linear Regression model to estimate the missing data
    model = linear_model.LinearRegression()
    model.fit(X = cukrzyca[parameters], y = cukrzyca[feature + '_imp'])
    
    #observe that I preserve the index of the missing data from the original dataframe
    deter_data.loc[cukrzyca[feature].isnull(), "Det" + feature] = model.predict(cukrzyca[parameters])[cukrzyca[feature].isnull()]

import matplotlib.pyplot as plt
sns.set()
fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.set_size_inches(8, 8)

for index, variable in enumerate(["Insulin", "SkinThickness"]):
    sns.distplot(cukrzyca[variable].dropna(), kde = False, ax = axes[index, 0])
    sns.distplot(deter_data["Det" + variable], kde = False, ax = axes[index, 0], color = 'red')
    
    sns.boxplot(data = pd.concat([cukrzyca[variable], deter_data["Det" + variable]], axis = 1),
                ax = axes[index, 1])
    
plt.tight_layout()

pd.concat([cukrzyca[["Insulin", "SkinThickness"]], deter_data[["DetInsulin", "DetSkinThickness"]]], axis = 1).describe().T

random_data = pd.DataFrame(columns = ["Ran" + name for name in missing_columns])

for feature in missing_columns:
        
    random_data["Ran" + feature] = cukrzyca[feature + '_imp']
    parameters = list(set(cukrzyca.columns) - set(missing_columns) - {feature + '_imp'})
    
    model = linear_model.LinearRegression()
    model.fit(X = cukrzyca[parameters], y = cukrzyca[feature + '_imp'])
    
    #Standard Error of the regression estimates is equal to std() of the errors of each estimates
    predict = model.predict(cukrzyca[parameters])
    std_error = (predict[cukrzyca[feature].notnull()] - cukrzyca.loc[cukrzyca[feature].notnull(), feature + '_imp']).std()
    
    #observe that I preserve the index of the missing data from the original dataframe
    random_predict = np.random.normal(size = cukrzyca[feature].shape[0], 
                                      loc = predict, 
                                      scale = std_error)
    random_data.loc[(cukrzyca[feature].isnull()) & (random_predict > 0), "Ran" + feature] = random_predict[(cukrzyca[feature].isnull()) & 
                                                                            (random_predict > 0)]
    
sns.set()
fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.set_size_inches(8, 8)

for index, variable in enumerate(["Insulin", "SkinThickness"]):
    sns.distplot(cukrzyca[variable].dropna(), kde = False, ax = axes[index, 0])
    sns.distplot(random_data["Ran" + variable], kde = False, ax = axes[index, 0], color = 'red')
    axes[index, 0].set(xlabel = variable + " / " + variable + '_imp')
    
    sns.boxplot(data = pd.concat([cukrzyca[variable], random_data["Ran" + variable]], axis = 1),
                ax = axes[index, 1])
    
    plt.tight_layout()

