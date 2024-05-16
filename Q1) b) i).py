# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:48:24 2023

@author: 2280933
"""

import pandas as pd
import pulp as pl

# Load the data
data = pd.read_excel('Q1.xlsx')

# Calculate returns
returns = data.pct_change().dropna()
assets = returns.loc[:, 'A1':'A6']
mean_returns = assets.mean()

# Find the covariance matrix
cov_matrix = assets.cov()

print("Covariance Matrix:")
print(cov_matrix)

# Get top 3 assets based on mean returns
top_assets3 = mean_returns.nlargest(3).index.tolist()

# Define the problem as a maximization problem for top 3 assets
prob_3 = pl.LpProblem('Asset Optimization (Top 3)', pl.LpMaximize)

# Define the decision variables for top 3 assets
x3 = {}
for asset in top_assets3:
    x3[asset] = pl.LpVariable(f'x3_{asset}', cat='Binary')

# Define the objective function for top 3 assets
prob_3 += pl.lpSum([mean_returns[asset]*x3[asset] for asset in top_assets3])

# Define the constraints for top 3 assets
prob_3 += pl.lpSum([x3[asset] for asset in top_assets3]) <= 1
prob_3 += pl.lpSum([mean_returns[asset]*x3[asset] for asset in top_assets3]) >= 0

# Solve the problem for top 3 assets
prob_3.solve()

# Print the optimal solution for top 3 assets
print('Optimal Solution (Top 3):')
for asset in top_assets3:
    print(f'{asset}: {pl.value(x3[asset])}')
print('Objective Value (Top 3) =', pl.value(prob_3.objective))

# Get top 4 assets based on mean returns
top_assets4 = mean_returns.nlargest(4).index.tolist()

# Define the problem as a maximization problem for top 4 assets
prob_4 = pl.LpProblem('Asset Optimization (Top 4)', pl.LpMaximize)
assets = returns.loc[:, 'A1':'A6']
# Define the decision variables for top 4 assets
x4 = {}
for asset in top_assets4:
    x4[asset] = pl.LpVariable(f'x4_{asset}', cat='Binary')

# Define the objective function for top 4 assets
prob_4 += pl.lpSum([mean_returns[asset]*x4[asset] for asset in top_assets4])

# Define the constraints for top 4 assets
prob_4 += pl.lpSum([x4[asset] for asset in top_assets4]) <= 1
prob_4 += pl.lpSum([mean_returns[asset]*x4[asset] for asset in top_assets4]) >= 0

# Solve the problem for top 4 assets
prob_4.solve()

# Print the optimal solution for top 4 assets
print('Optimal Solution (Top 4):')
for asset in top_assets4:
    print(f'{asset}: {pl.value(x4[asset])}')
print('Objective Value (Top 4) =', pl.value(prob_4.objective))
