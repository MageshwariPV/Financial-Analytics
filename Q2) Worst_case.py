# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:24:35 2023

@author: 2280933
"""

# Import the necessary packages
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from the excel file
data = pd.read_excel("Q2.xlsx", index_col=None, skiprows = [101], usecols="B:F")

# Initaialising the current and benchmark weights
current_weights = np.array([0.28, 0.02, 0.25, 0.1, 0.35])
benchmark_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

#Initialising the alpha and transaction cost values
alpha = 1
cost = 0.01

# splitting data into two subsets 
data_1 = data.sample(frac = 0.5, random_state=1)
data_2 = data.drop(data_1.index)

#Finding the mean value and covariance matrix of each dataset
returns = data.mean()
returns_1 = data_1.mean()
returns_2 = data_2.mean()

cov_matrix = data.cov()
cov_matrix_1 = data_1.cov()
cov_matrix_2 = data_2.cov()

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
        

# Return constraint for 100 dataset
def returns_constraint_all(x):
    weights  = x[:5]
    gamma = x[5]
    return np.sum(returns * weights) - gamma

# Risk constraint for 100 dataset
def risk_constraint_all(x):
    weights  = x[:5]
    lambd = x[6]
    return lambd - np.dot(weights.T, np.dot(cov_matrix, weights))

# Return constraint for first 50 dataset
def returns_constraint_sub1(x):
    weights  = x[:5]
    gamma = x[5]
    return np.sum(returns_1 * weights) - gamma

# Risk constraint for first 50 dataset
def risk_constraint_sub1(x):
    weights  = x[:5]
    lambd = x[6]
    return lambd - np.dot(weights.T, np.dot(cov_matrix_1, weights))

# Return constraint for next 50 dataset
def returns_constraint_sub2(x):
    weights  = x[:5]
    gamma = x[5]
    return np.sum(returns_2 * weights) - gamma

# Risk constraint for next 50 dataset
def risk_constraint_sub2(x):
    weights  = x[:5]
    lambd = x[6]
    return lambd - np.dot(weights.T, np.dot(cov_matrix_2, weights))

# Sum of weights constraint
def max_weight_constraint(x):
    weights  = x[:5]
    return np.sum(weights) - 1

# Benchmark constraint
def benchmark_constraint(x):
    weights = x[:5]
    return np.sum(returns * weights) - np.sum(returns * benchmark_weights)

# Return constraints
return_con = {'type': 'ineq', 'fun': returns_constraint_all}
return_con_sub1 = {'type': 'ineq', 'fun': returns_constraint_sub1}
return_con_sub2 = {'type': 'ineq', 'fun': returns_constraint_sub2}

# Risk constraints
risk_con = {'type': 'ineq', 'fun': risk_constraint_all}
risk_con_sub1 = {'type': 'ineq', 'fun': risk_constraint_sub1}
risk_con_sub2 = {'type': 'ineq', 'fun': risk_constraint_sub2}

# weight and beanchmarks
max_weight_con = {'type': 'eq', 'fun': max_weight_constraint}
benchmark_con = {'type': 'ineq', 'fun': benchmark_constraint}

cons = [return_con, return_con_sub1, return_con_sub2,
        risk_con, risk_con_sub1, risk_con_sub2,
        max_weight_con, benchmark_con]

b = (0,1)

bounds = [b, b, b, b, b,(-np.inf, np.inf), (-np.inf, np.inf)]

# Finding the solution
solution = minimize(objective_function, x0,
                    method = "SLSQP",
                    bounds = bounds,
                    constraints = cons)
print(solution)

#Looping for 10 values of alpha
alpha_values = np.linspace(0, 1, 10)
results = pd.DataFrame(columns=['alpha', 'obj_value', 'x'])
for alpha in alpha_values:
    cons[7]['fun'] = lambda x: np.sum(returns * x[:5]) - np.sum(returns * benchmark_weights) + alpha * (np.sum(abs(x[:5] - current_weights) * cost) - solution.fun)
    solution = minimize(objective_function, x0,
                         method = "SLSQP",
                         bounds = bounds,
                         constraints = cons)
    row = {'alpha': alpha,
           'obj_value': -solution.fun,
           'x': solution.x}
    results = results.append(row, ignore_index=True)
    
#Printing the results
print(results)
