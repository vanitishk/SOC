from asyncio import Task

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#Reading the csv file and loading the data
df = pd.read_csv("air_quality_data.csv")
X = df.drop("AQI", axis =1)
y = df["AQI"]

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

# --------------------------------------------------------------------------
print ("Task 1 starts now\n")

""" TASK 1 - FIT ORDINALRY LINEAR REGRESSION """
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
w = model.coef_
w0 = model.intercept_
print("Coefficients:", w)
print("Intercept:", w0)
for name, coef in zip(X_train.columns, w):
    print(name, "----", coef,)
print('\n')
print('\n')


# applying standardization to the features and fitting the model again
# it reduces the scale of features and can help with convergence 
# and interpretability of coefficients
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols_scaled_new = LinearRegression()
ols_scaled_new.fit(X_train_scaled, y_train)
for name, coef in zip(X_train.columns, ols_scaled_new.coef_):
    print(name, coef)
print("Intercept:", ols_scaled_new.intercept_)

print("Task 2 starts now\n")

#==============================================================================
#=============================================================================



""" Task 2 Ridge Regression Without Normalization"""
from sklearn.linear_model import Ridge

# lambda( ridge regression strenght given in question to evaluate coefficients 
# for all)
lambdas = [0.01, 0.1, 1, 10, 100, 1000]

#fitting the ridge regression model for each lambda value
for lam in lambdas:
    model = Ridge(alpha=lam)
    model.fit(X_train, y_train)

#printing the coefficients for each lambda value
    print("\nLambda =", lam)
    for name, coef in zip(X_train.columns, model.coef_):
        print(name, coef)

# Plot coefficient values vs log10(λ)
import matplotlib.pyplot as plt
coefs = []
for lam in lambdas:
    model = Ridge(alpha=lam)
    model.fit(X_train, y_train)
    coefs.append(model.coef_)
coefs = np.array(coefs)           #shape becomes (6,8) where 6 is number of lambda values and 8 is number of features

 #plotting the coefficients for each feature
for i in range(len(X_train.columns)):
    plt.plot(
        np.log(lambdas),
        coefs[:, i],
        label = X_train.columns[i]
    )                             
plt.xlabel("log(lambda)")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regression Coefficients vs Regularization Strength")
plt.legend()
plt.show(block =False)

"""" questions -
    1. As lambda increases, the coefficients shrink towards zero, indicating stronger regularization.
       Humidity, Wind Speed and Temperature shrink toward zero the fastest.
       Humidity ≈ -0.004   ( #lambda = 1000)
       Wind ≈ -0.009       ( #lambda = 1000)
       Temperature ≈ -0.010 ( #lambda = 1000)
    The coefficients for Humidity, Wind Speed and Temperature shrink towards zero the fastest 
    because they have smaller magnitudes compared to the other features.
    
    2.Which coefficient is most resistant to shrinkage? Now look for the coefficient 
    that remains large even under huge penalty.
    Ans -- co = 0.5 for PM2.5 is the most resistant to shrinkage as it remains 
    relatively large even as lambda increases, indicating that it has a strong 
    influence on the target variable (AQI) and is less affected by regularization 
    compared to the other features.
    
    This huge coefficient is partly caused by: different measurement scale 
    not necessarily because CO is genuinely more important.
    This is exactly what Researcher B was arguing.
    
    3. here a relationship between feature scale and shrinkage ---
    the y-axis stretches to accommodate CO.The other seven lines become compressed near zero.
    This tells us -
    Raw feature scales are affecting coefficient magnitudes.
    -------  Ridge penalizes coefficient size, NOT feature scale -------
    
        EX- FEATURE A  have values around 300, B has 0.5 ,
        to have same effect on prediction we need to have
        w1 = 0.5 w2 = 20
        ridge sees it as 0.5^2 and 20^2 ,so penalize B heavily

    Yes, there is a strong relationship between feature scale and shrinkage.
    """
print ("task 3 starts now\n")


#=============================================================================
#=============================================================================


"" " Task 3 - Ridge Regression with Normalization before it"""

#standardizing features ON TRAINING SET with z score (STANDARD SCALER)
#X_train_scaled = (X_train - X_train.mean()) / X_train.std()
#X_test_scaled = (X_test - X_train.mean()) / X_train.std()
# ( already done in task 1)
for lam in lambdas:
    model = Ridge(alpha=lam)
    model.fit(X_train_scaled, y_train)
    print("\nLambda =", lam)
    for name, coef in zip(X_train.columns, model.coef_):
        print(name, coef)


coeffs_task3 = []

for lam in lambdas:
    model = Ridge(alpha=lam)
    model.fit(X_train_scaled, y_train)
    coeffs_task3.append(model.coef_)
coeffs_task3 = np.array(coeffs_task3)

plt.figure(figsize=(10, 6))
for i in range(len(X_train.columns)):
    plt.plot(
        np.log10(lambdas),
        coeffs_task3[:, i],
        label = X_train.columns[i]
    )

plt.xlabel("log(lambda)")
plt.ylabel("Coefficient Value after Standardization")
plt.title("Ridge Regression Coefficients vs Regularization Strength, (Standardized first)")
plt.legend()
print("\ntask 3 done")
#plt.show()


#Reasearcher B is correct ---
 # Standardization dramatically changed coefficient magnitudes.
 # Features became comparable.
 # Ridge behaved more uniformly.
 # Interpretation became much easier.
 # The coefficient paths reveal the effect of regularization more clearly.

''' 
    Task 2 conclusion -
     Without normalization, coefficient magnitudes are heavily influenced by feature scale. 
     CO_ppm dominates the coefficient path and causes other features to appear compressed. 
     This makes coefficient interpretation difficult and demonstrates that Ridge regression is sensitive to feature scaling.

    Task 3 conclusion --
     After standardization, coefficient paths become more balanced and comparable. 
     Regularization affects all features more fairly, making interpretation easier. 
     These results show that feature normalization should be performed before applying Ridge regression. 
'''


# ==============================================================================


''' TASK 4 -- Lasso on Raw Data '''

from sklearn.linear_model import Lasso

lasso_coeff = []
for lam in lambdas:
    las = Lasso(alpha=lam)
    las.fit(X_train, y_train)
    print("\nLambdas :", lam)

    for name, coef in zip(X_train.columns , las.coef_): # to print feature name and its corresponding coefficient for each lambda value
     print(name, coef)
    
    lasso_coeff.append(las.coef_)
lasso_coeff = np.array(lasso_coeff)
print(lasso_coeff.shape)

#plotting lasso graph for each feature
plt.figure(figsize = (10,6))           
for i in range(len(X_train.columns)):
    plt.plot(
        #x axis , y axis
        np.log10(lambdas),lasso_coeff[ :,i ],
        label=X_train.columns[i]               #to label each line with feature name
    )
plt.xlabel("log10(lambda)")
plt.ylabel("Coefficient after applying Lasso L1")
plt.title("Lasso on raw data")
plt.legend()                                   #to make box with feature names
#plt.show()

# -----------------------  insights --------------------------
# Lasso feature entry is usually interpreted as:
# λ = 1000  → strongest penalty
# λ = 0.01  → weakest penalty
# Think of Lasso as asking: If I can keep only the strongest features,
# which ones survive first?

# At λ = 1000: Everything removed.
# At λ = 100: PM25, NO2 survive first. These are the most valuable features according to Lasso.

''' IT SUGGESTS =================
Large Ridge coefficient ≠ most important feature. 
Its large coefficient was partly due to scale effects.'''

#As λ increases, Lasso forces more coefficients to become exactly zero, performing automatic feature selection.
#PM25_ugm3 and NO2_ppb are the first features to enter the model, indicating that they provide the strongest predictive signal under heavy regularization.
#Unlike Ridge, Lasso produces sparse models by eliminating features completely.
#No feature is permanently excluded by Lasso. All eight features enter the model when λ becomes sufficiently small.

#PRINTING THE FEATURE TABLE 
for i , features in enumerate(X_train.columns):
    lambdas = [1000, 100, 10, 1, 0.1, 0.01]      #lasso penalty is so strong for large lambda
                                                 #that all coefficient become 0, and dec as lambda dec.
    #again fitting lasso for reverse oreder of lambda to match coefficient
    lasso_coeff = []
    for lam in lambdas:
        las = Lasso(alpha=lam)
        las.fit(X_train, y_train)
        lasso_coeff.append(las.coef_)
    lasso_coeff = np.array(lasso_coeff)
    
    non_zero = np.where((np.abs(lasso_coeff[:, i])> 10**-15))
    if len(non_zero)>0: 
        first_non_zero = non_zero[0][0]

        print(f"Features: {features} enters at lambda value {lambdas[first_non_zero]}")
    else:
        print(f"Feature: {features} never enters the model")
    




#==============================================================================
''' TASK 5 -- Lasso on Standardized Data '''
#=============================================================================

# X_train_scaled and y_train_scaled already computed in task 1
lasso_coeff_scaled = []
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
for lam in lambdas:
        las = Lasso(alpha=lam)
        las.fit(X_train_scaled, y_train)
        lasso_coeff_scaled.append(las.coef_)
lasso_coeff_scaled = np.array(lasso_coeff_scaled)
plt.figure(figsize=(10, 6))
for i in range(len(X_train.columns)):
    plt.plot(
        np.log10(lambdas),
        lasso_coeff_scaled[:, i],
        label=X_train.columns[i]
    )
plt.xlabel("log10(lambda)")
plt.ylabel("Coefficient after applying Lasso L1 (on Standardized X)")
plt.title("Lasso on standardized data")
plt.legend()
plt.show()