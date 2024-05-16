# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:24:35 2023

@author: 2280933
"""

#Import the necessary packages
import numpy as np
from scipy.optimize import minimize
import pandas as pd

## The dataset with 100 datapoints is considered for Scenario 1
#Reading the data from the excel file
data = pd.read_excel("Q2.xlsx", index_col=None, skiprows = [101], usecols="B:F")

# Initaialising the current and benchmark weights
current_weights = np.array([0.28, 0.02, 0.25, 0.1, 0.35])
benchmark_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

#Initialising the alpha and transaction cost values
alpha = 1
cost = 0.01

#Finding the mean and covariance matrix of the dataset
rates = data.mean()
cov_matrix = data.cov()

assets = data.columns
results = pd.DataFrame()

x0 = np.zeros(7)

# objective function
def objective_function(x, alpha = 1):
    weights  = x[:5]
    gamma = x[5]
    lambd = x[6]
    total_cost = np.sum(abs(weights - current_weights) * cost)
    obj_val = (alpha * (gamma - total_cost)) - ((1-alpha) * lambd)
    return -obj_val
        
#Return constraint
def returns_constraint_all(x):
    weights  = x[:5]
    gamma = x[5]
    return np.sum(rates * weights) - gamma

# Risk constraint 
def risk_constraint_all(x):
    weights  = x[:5]
    lambd = x[6]
    return lambd - np.dot(weights.T, np.dot(cov_matrix, weights))

# sum of weights constraint
def max_weight_constraint(x):
    weights  = x[:5]
    return np.sum(weights) - 1

# benchmark constraint
def benchmark_constraint(x):
    weights = x[:5]
    return np.sum(rates * weights) - np.sum(rates * benchmark_weights)

# return constraints
return_con = {'type': 'ineq', 'fun': returns_constraint_all}

# risk constraints
risk_con = {'type': 'ineq', 'fun': risk_constraint_all}

max_weight_con = {'type': 'eq', 'fun': max_weight_constraint}
benchmark_con = {'type': 'ineq', 'fun': benchmark_constraint}

cons = [return_con, risk_con, max_weight_con, benchmark_con]

b = (0,1)

bounds = [b, b, b, b, b,(-np.inf, np.inf), (-np.inf, np.inf)]

solution = minimize(objective_function, x0,
                    method = "SLSQP",
                    bounds = bounds,
                    constraints = cons)
print(solution)

#Looping through 10 alpha values
alpha_values = np.linspace(0, 1, 10)
results = pd.DataFrame(columns=['alpha', 'obj_value', 'x'])
for alpha in alpha_values:
    cons[3]['fun'] = lambda x: np.sum(rates * x[:5]) - np.sum(rates * benchmark_weights) + alpha * (np.sum(abs(x[:5] - current_weights) * cost) - solution.fun)
    solution = minimize(objective_function, x0,
                         method = "SLSQP",
                         bounds = bounds,
                         constraints = cons)
    row = {'alpha': alpha,
           'obj_value': -solution.fun,
           'x': solution.x}
    results = results.append(row, ignore_index=True)
print(results)
