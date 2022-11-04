import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
from sklearn.preprocessing import StandardScaler
set_style('whitegrid')
import sys
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

houses_train = pd.read_csv('./train.csv', index_col='Id')
houses_test = pd.read_csv('test.csv')

# Cleaning float columns
nan_cols = houses_train.loc[:, houses_train.isna().any()].columns
nan_cols_test = houses_test.loc[:, houses_test.isna().any()].columns
nan_cols_num = [col for col in nan_cols if houses_train[col].dtype == float]
nan_cols_num_test = [col for col in nan_cols_test if houses_test[col].dtype == float]
impute = SimpleImputer()
impute.fit(houses_train[nan_cols_num])
houses_train[nan_cols_num] = impute.transform(houses_train[nan_cols_num])
houses_test[nan_cols_num] = impute.transform(houses_test[nan_cols_num])
impute.fit(houses_train[nan_cols_num_test])
houses_test[nan_cols_num_test] = impute.transform(houses_test[nan_cols_num_test])

# Cleaning categorical columns
cat_cols = houses_train.select_dtypes(include=[object]).columns
for col in cat_cols:
    houses_train = houses_train.join(pd.get_dummies(houses_train[col], dummy_na=True, prefix=col))
    houses_train.drop(col, axis=1, inplace=True)

    houses_test = houses_test.join(pd.get_dummies(houses_test[col], dummy_na=True, prefix=col))
    houses_test.drop(col, axis=1, inplace=True)


"""
corrs = houses_train.corr()['SalePrice']
corrs = corrs.sort_values()
print(corrs.iloc[-6 : ])
"""

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
houses_train['GarageCarsxArea'] = houses_train['GarageCars'] * houses_train['GarageArea']
houses_test['GarageCarsxArea'] = houses_test['GarageCars'] * houses_test['GarageArea']
allFeatures = list(houses_train.columns)
allFeatures.remove('SalePrice')

# Add in columns to test set that only appear in training set
allFeaturesSet = set(allFeatures)
for feature in allFeaturesSet - set(list(houses_test.columns)):
    houses_test[feature] = np.zeros(len(houses_test))

"""
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88))])
pipe.fit(houses_train[allFeatures].values)
houses_train_pca = pd.DataFrame(pipe.transform(houses_train[allFeatures].values))
houses_test_pca = pd.DataFrame(pipe.transform(houses_test[allFeatures].values))
houses_train_pca['SalePrice'] = houses_train['SalePrice']
"""

kfold = KFold(5, shuffle=True)
models = [
        #'baseline',
        #'knn',
        #'reg',
        #[79, 95, 25]
        #features,
        #features + ['GarageCarsxArea'],
        #features + ['1stFlrSF'],
        #features + ['GarageCarsxArea', '1stFlrSF'],
        #'pca',
        #'pca_knn',
        #'pca_quad',
        #'pca_forest'
        ]
cv_mses = np.zeros((5, len(models)))

"""
for i, (train_idx, test_idx) in enumerate(kfold.split(houses_train_pca)):
    houses_tt = houses_train.iloc[train_idx]
    houses_ho = houses_train.iloc[test_idx]

    for j, model in enumerate(models):
        if model == 'baseline':
            pred = houses_tt['SalePrice'].mean() * np.ones(len(houses_ho))
        elif model == 'knn':
            knr = KNeighborsRegressor(10)
            knr.fit(houses_tt.loc[:, houses_tt.columns != 'SalePrice'].values, houses_tt['SalePrice'].values)
            pred = knr.predict(houses_ho.loc[:, houses_ho.columns != 'SalePrice'].values)
        elif model == 'pca':
            pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('reg', LinearRegression())])
            pipe.fit(houses_tt[allFeatures].values, houses_tt['SalePrice'].values)
            pred = pipe.predict(houses_ho[allFeatures].values)
        else:
            reg = LinearRegression()
            reg.fit(houses_tt.loc[:, houses_tt.columns != 'SalePrice'].values, houses_tt['SalePrice'].values)
            pred = reg.predict(houses_ho.loc[:, houses_ho.columns != 'SalePrice'].values)

        cv_mses[i, j] = mean_squared_error(houses_ho['SalePrice'], pred)
"""


for i, (train_idx, test_idx) in enumerate(kfold.split(houses_train)):
    houses_tt = houses_train.iloc[train_idx]
    houses_ho = houses_train.iloc[test_idx]

    for j, model in enumerate(models):
        if model == 'baseline':
            pred = houses_tt['SalePrice'].mean() * np.ones(len(houses_ho))
        elif model == 'knn':
            knr = KNeighborsRegressor(10)
            knr.fit(houses_tt[features].values, houses_tt['SalePrice'].values)
            pred = knr.predict(houses_ho[features].values)
        elif model == 'pca':
            pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('reg', LinearRegression())])
            pipe.fit(houses_tt[allFeatures].values, houses_tt['SalePrice'].values)
            pred = pipe.predict(houses_ho[allFeatures].values)
        elif model == 'pca_knn':
            pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('knr', KNeighborsRegressor(10))])
            pipe.fit(houses_tt[allFeatures].values, houses_tt['SalePrice'].values)
            pred = pipe.predict(houses_ho[allFeatures].values)
        elif model == 'pca_quad':
            pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('poly', PolynomialFeatures(2, include_bias=False)), ('reg', LinearRegression())])
            pipe.fit(houses_tt[allFeatures].values, houses_tt['SalePrice'].values)
            pred = pipe.predict(houses_ho[allFeatures].values)
        elif model == 'pca_forest':
            pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('forest', RandomForestRegressor())])
            pipe.fit(houses_tt[allFeatures].values, houses_tt['SalePrice'].values)
            pred = pipe.predict(houses_ho[allFeatures].values)
        else:
            reg = LinearRegression()
            reg.fit(houses_tt[model].values, houses_tt['SalePrice'].values)
            pred = reg.predict(houses_ho[model].values)

        cv_mses[i, j] = mean_squared_error(houses_ho['SalePrice'], pred)

print(np.mean(cv_mses, axis=0))

pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(0.88)), ('forest', RandomForestRegressor())])
pipe.fit(houses_train[allFeatures].values, houses_train['SalePrice'].values)
pred = pipe.predict(houses_test[allFeatures].values)

results = pd.DataFrame({'Id': houses_test['Id'], 'SalePrice': pred})
results = results.set_index('Id')
results.to_csv('submission.csv')