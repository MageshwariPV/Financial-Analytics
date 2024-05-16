# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 00:01:23 2023

@author: 2280933
"""
#Importing the necessary packages
import matplotlib.pyplot as plt

# Define the scenario returns and risks
scenario1_returns = [0.7334, 0.7353, 0.7384, 0.7426, 0.7459, 0.7507, 0.7570, 0.7607, 0.7652, 0.7695]
scenario2_returns = [0.7461, 0.7480, 0.7550, 0.7561, 0.7594, 0.7648, 0.7681, 0.7732, 0.7731, 0.7743]
scenario3_returns = [0.7673, 0.7723, 0.7771, 0.7845, 0.7893, 0.7982, 0.8063, 0.8069, 0.8079, 0.8096]
worst_case_returns = [0.7325, 0.7348, 0.7376, 0.7406, 0.7435, 0.7455, 0.7490, 0.7599, 0.7679, 0.7679]

scenario1_risks = [0.0360, 0.0391, 0.0404, 0.0434, 0.0469, 0.0539, 0.0639, 0.0712, 0.0855, 0.2075]
scenario2_risks = [0.0307, 0.0313, 0.0368, 0.0380, 0.0429, 0.0494, 0.0576, 0.0797, 0.1127, 0.1615]
scenario3_risks = [0.0426, 0.0474, 0.0508, 0.0560, 0.0626, 0.0732, 0.0849, 0.0862, 0.0929, 0.1295]
worst_case_risks = [0.0453, 0.0457, 0.0468, 0.0487, 0.0517, 0.0550, 0.0605, 0.0884, 0.1173, 0.2253]

# Create the plot
plt.plot(scenario1_risks, scenario1_returns, label='Scenario 1')
plt.plot(scenario2_risks, scenario2_returns, label='Scenario 2')
plt.plot(scenario3_risks, scenario3_returns, label='Scenario 3')
plt.plot(worst_case_risks, worst_case_returns, label='Worst Case Scenario')

# Add axis labels and a title
plt.xlabel('Risks(%)')
plt.ylabel('Returns(%)')
plt.title('Efficient Frontier for Worst Case Analysis and Three risk-rival scenarios ')

# Add a legend
plt.legend()

# Display the plot
plt.show()
