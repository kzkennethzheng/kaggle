import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('whitegrid')
import sys

houses_train = pd.read_csv('train.csv', index_col='Id')

for column in houses_train.columns:
    if houses_train[column].isnull().values.any():
        continue

    plt.figure(figsize=(14, 4))

    plt.scatter(houses_train[column], houses_train['SalePrice'])
    plt.title(column)
    plt.show()
    sys.stdin.readline()

corrs = houses_train.corr()['SalePrice']
corrs = corrs.sort_values()
corrs.iloc[-6 : ]

#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
"""
LotArea
Street, MSZoning, #cats
TotalBsmtSF    0.613581
GarageArea     0.623431
GarageCars     0.640409 #freq count
GrLivArea      0.708624
OverallQual    0.790982

models: baseline mean, regression on 5 features, regression on 5 with some interaction, dimension reduction
"""
