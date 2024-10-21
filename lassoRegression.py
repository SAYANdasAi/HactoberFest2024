# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate some random data for illustration
# Independent variable (X)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Dependent variable (y) with some noise
y = np.array([2, 4, 5, 4, 5, 6, 7, 8, 9, 10])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Ridge and Lasso regression models with regularization parameter alpha=1.0
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)

# Train the models
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Make predictions using the testing set
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)

# Print Ridge regression coefficients
print("Ridge Regression")
print(f"Intercept: {ridge_model.intercept_}")
print(f"Coefficient: {ridge_model.coef_[0]}")
print(f"Mean Squared Error (Ridge): {mean_squared_error(y_test, ridge_pred)}")

# Print Lasso regression coefficients
print("\nLasso Regression")
print(f"Intercept: {lasso_model.intercept_}")
print(f"Coefficient: {lasso_model.coef_[0]}")
print(f"Mean Squared Error (Lasso): {mean_squared_error(y_test, lasso_pred)}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual Data')

# Ridge regression line
plt.plot(X_test, ridge_pred, color='red', label='Ridge Prediction')

# Lasso regression line
plt.plot(X_test, lasso_pred, color='green', label='Lasso Prediction')

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
