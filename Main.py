import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_csv("housing.csv").iloc[:5000]


# Split features and target
X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]

# Identify numerical and categorical features
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']

# Apply scaling to numerical features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=X.index)  # Ensure index alignment
X = X.drop(columns=categorical_features)
X = pd.concat([X, encoded_df], axis=1)

# Handle missing values in numerical features
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Data Visualization
def plot_histograms(data):
    n_features = min(data.shape[1], 5)  # Limit to the first 5 columns if more than 5 features
    n_rows = (n_features + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, n_rows * 3))
    axs = axs.ravel()
    for i in range(n_features):
        axs[i].hist(data.iloc[:, i], bins=20, alpha=0.75)
        axs[i].set_title(data.columns[i])
    # Hide any unused subplots
    for j in range(n_features, n_rows * 2):
        axs[j].set_visible(False)
    plt.tight_layout()
    plt.show()

# Plot histograms for the first 5 columns
plot_histograms(X.iloc[:, :5])

def plot_2d(data, x_col, y_col):
    plt.scatter(data[x_col], data[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.show()

def plot_3d(data, x_col, y_col, z_col):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x_col], data[y_col], data[z_col], c='b', marker='o')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title(f"{x_col}, {y_col}, {z_col}")
    plt.show()

plot_histograms(X)
plot_2d(X, 'longitude', 'latitude')
plot_3d(X, 'longitude', 'latitude', 'housing_median_age')

# Regression Models
def evaluate_model(model, X, y):
    mse = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True), scoring='neg_mean_squared_error')
    mae = cross_val_score(model, X, y, cv=KFold(n_splits=10, shuffle=True), scoring='neg_mean_absolute_error')
    return np.sqrt(-mse), -mae

# Perceptron Algorithm
perceptron_model = Perceptron()
perceptron_rmse, perceptron_mae = evaluate_model(perceptron_model, X, y)
print("Perceptron RMSE:", perceptron_rmse.mean())
print("Perceptron MAE:", perceptron_mae.mean())

# Least Squares Algorithm (Linear Regression)
linear_regression_model = LinearRegression()
linear_regression_rmse, linear_regression_mae = evaluate_model(linear_regression_model, X, y)
print("Linear Regression RMSE:", linear_regression_rmse.mean())
print("Linear Regression MAE:", linear_regression_mae.mean())

# Multilayer Neural Network
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=50)
mlp_rmse, mlp_mae = evaluate_model(mlp_model, X, y)
print("MLP RMSE:", mlp_rmse.mean())
print("MLP MAE:", mlp_mae.mean())