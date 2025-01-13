import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Custom Perceptron Model
class CustomPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Custom Linear Regression Model
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            d_weights = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            d_bias = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * d_weights
            self.bias -= self.lr * d_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class SimpleMLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, 1)
        self.bias_output = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        for _ in range(self.epochs):
            # Forward pass
            hidden_layer_input = X.dot(self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = hidden_layer_output.dot(self.weights_hidden_output) + self.bias_output
            predicted_output = self.sigmoid(output_layer_input)

            # Backward pass
            error = y - predicted_output
            d_predicted_output = error * self.sigmoid_derivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
            self.bias_output += np.sum(d_predicted_output, axis=0) * self.learning_rate

            self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias_hidden += np.sum(d_hidden_layer, axis=0) * self.learning_rate

    def predict(self, X):
        hidden_layer_input = X.dot(self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = hidden_layer_output.dot(self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)
        return np.where(predicted_output >= 0.5, 1, 0)

# Load dataset
data = pd.read_csv("housing.csv").iloc[:2000]

# Split features and target
X = data.drop(columns=["median_house_value"])
y = data["median_house_value"]

# Convert target to binary
median_value = y.median()
y = np.where(y > median_value, 1, 0)

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
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=X.index)
X = X.drop(columns=categorical_features)
X = pd.concat([X, encoded_df], axis=1)

# Handle missing values in numerical features
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Convert to NumPy arrays with correct types
X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Perceptron model
perceptron = CustomPerceptron(learning_rate=0.01, epochs=1000)
perceptron.fit(X_train, y_train)

# Predict and evaluate on the test set
perceptron_predictions = perceptron.predict(X_test)
perceptron_accuracy = np.mean(perceptron_predictions == y_test)
print(f"Perceptron binary classification accuracy: {perceptron_accuracy:.4f}")

# Initialize and train the custom Linear Regression model
ls_lr = CustomLinearRegression(learning_rate=0.01, epochs=1000)
ls_lr.fit(X_train, y_train)

# Predict and evaluate on the test set
ls_lr_predictions = ls_lr.predict(X_test)
ls_lr_accuracy = np.mean(np.abs(ls_lr_predictions - y_test))
print(f"Least Squares Linear Regression MAE: {ls_lr_accuracy:.4f}")



# Initializing and training the MLP model
input_size = X_train.shape[1]
hidden_size = 10  # Example: 10 hidden neurons
mlp = SimpleMLP(input_size, hidden_size, learning_rate=0.01, epochs=1000)
mlp.fit(X_train, y_train)

# Predicting and evaluating the MLP model
mlp_predictions = mlp.predict(X_test)
mlp_accuracy = np.mean(mlp_predictions.flatten() == y_test)
print(f"MLP binary classification accuracy: {mlp_accuracy:.4f}")
