# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:55:43 2023

@author: 2280933
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_excel('Q1.xlsx')

# Calculate returns
returns = data.pct_change().dropna()

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Intercept', 'Factor Loading', 'Average Residual'])
assets = returns.loc[:, 'A1':'A6']
## For each asset
for col in assets.columns:
    # Extract the factors and asset returns
    X = np.array(returns['FACTOR 1']).reshape(-1,1)
    y = returns[col].values.reshape(-1, 1)

    # Fit the linear regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Calculate the intercept, factor loading, residual, and average residual
    intercept = model.intercept_[0]
    factor_loading = model.coef_[0][0]
    residual = y - model.predict(X)
    average_residual = residual.mean().item()

    # Add the results to the DataFrame
    results_df.loc[col] = [intercept, factor_loading, average_residual]

# Display the results
print(results_df)
