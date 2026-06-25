import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. Data Setup
# ==========================================
train_df = pd.read_csv('q5_train.csv')
test_df = pd.read_csv('q5_test.csv')

X_train = train_df.drop(columns=['y'])
y_train = train_df['y']
X_test = test_df.drop(columns=['Id'])
features = X_train.columns

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# TASK 1: Data Collection
# ==========================================
alphas = np.logspace(-4, 3, 100)
ridge_coefs, lasso_coefs = [], []

for a in alphas:
    # Ridge Model
    ridge = Ridge(alpha=a).fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    
    # Lasso Model
    lasso = Lasso(alpha=a, max_iter=10000).fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

# 1. & 2. Plot Coefficient Paths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for i, feature in enumerate(features):
    ax1.plot(alphas, np.array(ridge_coefs)[:, i], label=feature)
    ax2.plot(alphas, np.array(lasso_coefs)[:, i], label=feature)

ax1.set(xscale='log', title='Ridge Coefficient Paths', xlabel='Lambda', ylabel='Coefficient Magnitude')
ax2.set(xscale='log', title='Lasso Coefficient Paths', xlabel='Lambda')
ax1.legend(); ax2.legend()
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 3. Scatter Plots of features vs y (Helps with Task 2.1 and 2.5)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, feature in enumerate(features):
    row, col = divmod(i, 4)
    axes[row, col].scatter(train_df[feature], y_train, alpha=0.5)
    axes[row, col].set_title(f'{feature} vs y')
plt.tight_layout()
plt.show()

# ==========================================
# TASK 2: Forensic Analysis Helpers
# ==========================================

print("--- Task 2 Forensic Helpers ---")

# Task 2.2: Identify Correlated Features
# Finds the pair with the absolute highest correlation (excluding 'y')
X_corr = X_train.corr().abs()
np.fill_diagonal(X_corr.values, 0) # Ignore self-correlation
max_corr_idx = X_corr.unstack().argmax()
feat_a, feat_b = X_corr.unstack().index[max_corr_idx]
print(f"Task 2.2: The highly correlated 'grouping' pair is {feat_a} and {feat_b} with a correlation of {X_corr.loc[feat_a, feat_b]:.2f}")

# Task 2.3: Reliability vs Availability
# We generate a pairplot of all features against each other.
# Instructions: Look for a plot where the points form a diagonal line, but one feature has a dense line of dots dropping to exactly 0 or a constant.
print("Generating Pairplot for Task 2.3... (Look for intermittent dropouts in the feature-vs-feature grids)")
sns.pairplot(X_train, corner=True, plot_kws={'alpha':0.5, 's':10})
plt.suptitle("Task 2.3: Feature vs Feature (Identify Reliability vs Availability)", y=1.02)
plt.show()


# ==========================================
# TASK 2.6: Final Model Construction
# ==========================================
# Action Required: 
# 1. Look at the Lasso plot. Find the feature that drops to 0 instantly. Add it to the list below.
# 2. Look at the scatter plots. 'x8' clearly has extreme outliers (structural corruption).

noise_feature = 'x7' # Replace this with the actual feature you identify from the Lasso plot!
features_to_drop = ['x8', noise_feature] 

print(f"\n--- Final Model ---")
print(f"Dropping compromised features: {features_to_drop}")

X_train_clean = X_train.drop(columns=features_to_drop, errors='ignore')
X_test_clean = X_test.drop(columns=features_to_drop, errors='ignore')

# Re-scale cleaned data
scaler_clean = StandardScaler()
X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
X_test_clean_scaled = scaler_clean.transform(X_test_clean)

# Train optimal Ridge Model
final_model = RidgeCV(alphas=np.logspace(-2, 2, 100))
final_model.fit(X_train_clean_scaled, y_train)

print(f"Optimal Ridge Lambda: {final_model.alpha_:.4f}")

# Generate Predictions
predictions = final_model.predict(X_test_clean_scaled)
submission = pd.DataFrame({'Id': test_df['Id'], 'y': predictions})
submission.to_csv('submission.csv', index=False)
print("Successfully generated 'submission.csv' for Kaggle upload.")