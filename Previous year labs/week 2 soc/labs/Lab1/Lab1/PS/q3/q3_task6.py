import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('air_quality_data.csv')
# print(df) ===  print csv file with index from first cloumn value
# print(df.columns)  ===. print column names of the csv file in a list format 
# print(df.head()). === print first 5 rows of the csv file
# print(df.describe()). === print statistical summary of the csv file(count, mean, std, min, 25%, 50%, 75%, max)
# print(df.info()) ===  print concise summary of the DataFrame, including the number of non-null entries and data types of each column
# print(df.isnull().sum()) === print the number of missing values in each column of the


# ==================================================================
''' OUTLIERS IN PM2.5 AND CO '''
# ===================================================================

PM25_outliers = df[df['PM25_ugm3'] > 200]
CO_outliers = df[df['CO_ppm'] > 4]
# print("Outliers in PM2.5:" , PM25_outliers)
# print("\nOutliers in CO:" , CO_outliers)
print (len(PM25_outliers), "outliers in PM2.5")
print (len(CO_outliers), "outliers in CO")

'''Task 6 Analysis
1-Using the IQR method, 14 outliers were detected in PM2.5 and 9 outliers in CO.
2-Several observations appeared as outliers in both variables, indicating that extreme PM2.5 values are often associated with extreme CO concentrations.
3-Inspection of the corresponding AQI values shows that many of these observations also have unusually high AQI, suggesting that they likely represent genuine severe pollution events rather than data-entry errors.
4-Outliers can strongly influence Ordinary Least Squares regression because squared residuals give extreme observations disproportionately large influence.
5-Ridge ( Model becomes more stable than OLS as coeff are shrunk) and Lasso regularization reduce coefficient instability but do not completely eliminate the impact of outliers since both methods still use squared prediction errors.
Therefore, outlier detection is important before fitting regression models, especially when interpreting coefficients.'''

#unique outliers in PM2.5 and CO
all_outliers = set(PM25_outliers.index) | set(CO_outliers.index)
print("Total unique outliers:", len(all_outliers))

# creating a new DataFrame A with outliers removed
df_A = df.drop(index=all_outliers)

#creating testing and training data for A
X_A = df_A.drop('AQI', axis=1)
y_A = df_A['AQI']
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=5)

#normalising A through standardizing
scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

# creatin B  dataset without outliers
X_B = df.drop('AQI', axis=1)
y_B = df['AQI']
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=5)
    #----  standard scalar uses mean and sd which are affected by outliers, 
    #----  but here ROBUST SCALAING used median and IQR which are less affected by outliers
robust_scalar = RobustScaler()
X_train_B_scaled = robust_scalar.fit_transform(X_train_B)
X_test_B_scaled = robust_scalar.transform(X_test_B)

#Ridge on A
coeffs_A = []
lambdas = [0.01, 0.1, 1, 10, 100, 1000]
for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_A_scaled, y_train_A)

    coeffs_A.append(ridge.coef_)
coeffs_A = np.array(coeffs_A)

#Ridge on B
coeffs_B = []
for lam in lambdas:
    ridge = Ridge(alpha = lam)
    ridge.fit(X_train_B , y_train_B)
    coeffs_B.append(ridge.coef_)
coeffs_B = np.array(coeffs_B)

## A performance 
print("Dataset A Performance\n")
mse_A = []
r2_A = []

for lam in lambdas:

    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_A_scaled, y_train_A)

    y_pred = ridge.predict(X_test_A_scaled)

    mse_A.append(
        mean_squared_error(y_test_A, y_pred)
    )

    r2_A.append(
        r2_score(y_test_A, y_pred)
    )
print(mse_A)
print(r2_A)

## B performance
print("Dataset B Performance\n")
mse_B = []
r2_B = []

for lam in lambdas:

    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_B_scaled, y_train_B)

    y_pred = ridge.predict(X_test_B_scaled)

    mse_B.append(
        mean_squared_error(y_test_B, y_pred)
    )

    r2_B.append(
        r2_score(y_test_B, y_pred)
    )
print(mse_B)
print(r2_B)



## --- ploting A after ridge
for i in range(len(X_train_A.columns)):
    plt.plot(
        np.log10(lambdas),
        coeffs_A[: , i],
        label = X_train_A.columns[i]
    )
plt.xlabel("log10(lambda)")
plt.ylabel("coefficient with ridge (no outlier taken)")
plt.title("data set A- without PM25 and CO outlier")
plt.legend()

## ----- Ploting B after ridge
plt.figure(figsize=(10,6))
for i in range(len(X_train_B.columns)):
    plt.plot(
        np.log10(lambdas),
        coeffs_B[: , i],    
        label = X_train_B.columns[i]
    )
plt.xlabel("log10(lambda)")
plt.ylabel("coefficient after Ridge")
plt.title('effect of Robust scaling (with outliers)')
plt.legend()
plt.show()