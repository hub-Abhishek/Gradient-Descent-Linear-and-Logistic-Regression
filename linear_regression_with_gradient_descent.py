

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Defining functions for Linear Regression

#1. Calculate slope of the cost function
def calc_m_lin(m, alpha, x, y, length):
    return (m - alpha*(x.T @ (x @ m - y))/length)

#2. Calculate value of the cost function
def calc_cost_lin(m, x, y, length):
    return sum((x @ m - y)**2)/(2*length)

#3. Setting up the gradient descent algorithm
def grad_desc_lin(x, y, length, m, alpha, iters, costs, threshold = 0):
    
    for i in range(0,iters):
        cost = calc_cost_lin(m, x, y, length)[0]
        costs.append(cost)
        m = calc_m_lin(m, alpha, x, y, length)
        if i > 1 and (costs[len(costs) - 2] - costs[len(costs) - 1] <= threshold):
            break
    return [m, costs, i]

#4. Function to print model outputs
def model_parameters(coeff, x, y, names):
    
    err = np.subtract(y, x@coeff)
    #Calculating R2 = 1 - sse/sst
    sse = np.sum(err**2)
    sst = np.sum((y - np.mean(y, axis = 0))**2) 
    r2 = 1 - (sse/sst)
    
    #Calculating Adj R2 = 1 - (1 - r2)*(n - 1)/(n - k - 1)
    adj_r2 = 1 - (1 - r2)*(len(x) - 1)/(len(x) - len(coeff) - 1)
    
    #variable name, coeff, p val, t - stat, std errors
    model_output = pd.DataFrame(columns = ['Variable', 'Coeff', 'Std Err', 'T- Val'])
    df = len(x) - len(coeff) - 1
    x_terms = np.sum(np.subtract(x, np.mean(x, axis = 0))**2, axis = 0)
    for i in range(0, len(coeff)):
        se = np.sqrt(sse/df)/np.sqrt(x_terms[i] + np.finfo(float).eps)
        t  =coeff[i]/se
        model_output.loc[i] = [names[i], coeff[i][0], se, t[0]]
    print('Model Output:\n')
    print(model_output)
    print('\n')
    
    #Mean Absolute Error
    mae = np.sum(abs(err))/len(x)
    
    #Mean Squared Error
    mse = np.sum(err**2)/len(x)
    
    #Mean Absolute Percentage Error
    mape = 100 * np.sum(abs(1-err/y))/len(x)
    
    #Root Mean Squared Error
    rmse = np.sqrt(np.sum(err**2)/len(x))
    print("R2 of the estimated model: {}".format(r2))
    print("Adjusted R2 of the estimated model: {}".format(adj_r2))
    print('Mean Absolute Error: {}'.format(mae))
    print('Mean Squared Error: {}'.format(mse))
    print("Mean Absolute Percentage Error: {}%".format(mape))
    print("Root Mean Squared Error: {}".format(rmse))

# Implementing L1 and L2 norms for Linear regression

# To implement L1 and L2 norms, we just need to modify our cost function. We will add L1 and L2 weights to our cost function, leaving the other calculations intact.

#1. Calculate slope of the cost function
def calc_m_lin(m, alpha, x, y, length):
    return (m - alpha*(x.T @ (x @ m - y))/length)

#2. Calculate value of the cost function
def calc_cost_lin_reg(m, x, y, length, reg, lambdaa):
    err = sum((x @ m - y)**2)/(2*length)
    if reg == 'l1' or reg == 'lasso':
        return err + lambdaa*np.sum(np.absolute(m[1:]))
    elif reg == 'l2' or reg == 'ridge':
        return err + lambdaa*np.sum(m[1:]**2)    
    elif reg == None:
        return err

#3. Setting up the gradient descent algorithm
def grad_desc_lin_reg(x, y, length, m, alpha, iters, costs, threshold = 0, reg = None, lambdaa = 0.01):
    
    for i in range(0,iters):
        cost = calc_cost_lin_reg(m, x, y, length, reg, lambdaa)[0]
        costs.append(cost)
        m = calc_m_lin(m, alpha, x, y, length)
        if i > 1 and (costs[len(costs) - 2] - costs[len(costs) - 1] <= threshold):
            break
    return [m, costs, i]