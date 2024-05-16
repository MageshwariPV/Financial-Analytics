# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 12:19:17 2023

@author: 2280933
"""

import pandas as pd
import pulp as pl
from pulp import *

# Load the data
data = pd.read_excel('Q1.xlsx')

# Calculate returns
returns = data.pct_change().dropna()
assets = returns.loc[:, 'A1':'A6']
mean_returns = assets.mean()

# Create a new LP problem
prob = LpProblem("AssetAllocation", LpMaximize)

# Define the decision variables
x = LpVariable.dicts("x", assets, 0, 1, LpBinary)
w = LpVariable.dicts("w", assets, lowBound=0, upBound=0.2)

# Define the objective function
prob += lpSum([mean_returns[i] * w[i] - (w[i] * 0.015) for i in assets])

# Define the constraints
prob += lpSum([w[i] for i in assets]) <= 1
for i in assets:
    prob += w[i] <= x[i] 
    
#For 3 stocks portfolio
prob += lpSum([x[i] for i in assets]) <= 3

# Solve the LP problem
prob.solve()

# Print the optimal solution and the optimal objective value
print("Optimal Solution:")
for i in assets:
    print(f"{i}: {x[i].varValue:.0f}")
print(f"Optimal Objective Value: {value(prob.objective):.4f}")
print("Status:", pl.LpStatus[prob.status])

# For 4 stock portfolio

# Create a new LP problem
prob = LpProblem("AssetAllocation", LpMaximize)

# Define the decision variables
x = LpVariable.dicts("x", assets, 0, 1, LpBinary)
w = LpVariable.dicts("w", assets, lowBound=0, upBound=0.2)

# Define the objective function
prob += lpSum([mean_returns[i] * w[i] - (w[i] * 0.015) for i in assets])

# Define the constraints
prob += lpSum([w[i] for i in assets]) <= 1
for i in assets:
    prob += w[i] <= x[i] 
prob += lpSum([x[i] for i in assets]) <= 4

# Solve the LP problem
prob.solve()

# Print the optimal solution and the optimal objective value
print("Optimal Solution:")
for i in assets:
    print(f"{i}: {x[i].varValue:.0f}")
print(f"Optimal Objective Value: {value(prob.objective):.4f}")
print("Status:", pl.LpStatus[prob.status])
