import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from geneticalgorithm import geneticalgorithm as ga
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Measure Genetic Algorithm (GA) Runtime
start_time = time.time()

#df = pd.read_csv('insurance.csv').sample(1000, random_state=42)
data = pd.read_csv('hospital_readmissions.csv').sample(1000, random_state=42)

# Train-test split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Feature definitions
numeric_features = [
    "time_in_hospital", "n_procedures", "n_lab_procedures", "n_medications",
    "n_outpatient", "n_inpatient", "n_emergency"
]

binary_features = ["change", "diabetes_med"]
categorical_features = [
    "age", "medical_specialty", "diag_1", "diag_2", "diag_3",
    "glucose_test", "A1Ctest"
]
target = 'readmitted'

# Encode binary attributes
train_df[binary_features] = train_df[binary_features].apply(
    lambda x: x.map({ 'no': 0, 'yes': 1}))
test_df[binary_features] = test_df[binary_features].apply(lambda x: x.map({'no': 0, 'yes': 1}))


# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),
    ('bin', MinMaxScaler(), binary_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
])


# Transform data
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)

#print(X_train[:5])
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)


train_df[target] = train_df[target].map({'no': 0, 'yes': 1}).astype(float)
test_df[target] = test_df[target].map({'no': 0, 'yes': 1}).astype(float)

y_train = train_df[target].values
y_test = test_df[target].values


y_train = train_df[target].values
y_test = test_df[target].values


def calculate_similarities(X, weights, k=3):
    """
    Vectorized similarity calculation with k-NN support
    Returns top k similarities and indices
    """
    weighted_diffs = np.abs(X[:, None] - X) * weights
    similarities = -np.sum(weighted_diffs, axis=2)  # Negative Manhattan distance
    np.fill_diagonal(similarities, -np.inf)  # Ignore self-similarity
    top_k_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
    return top_k_indices


def fitness_function(weights):
    weights = np.abs(weights)  # Ensure non-negative weights
    k = 3  # Number of neighbors
    neighbors = calculate_similarities(X_train, weights, k)
    # Calculate predictions as mean of neighbors
    predictions = np.mean(y_train[neighbors], axis=1)
    return mean_absolute_error(y_train, predictions)


algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 50,
    'mutation_probability': 0.1,
    'elit_ratio': 0.1,
    'crossover_probability': 0.8,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 25,
    'convergence_curve': True,
}

model = ga(
    function=fitness_function,
    dimension=X_train.shape[1],
    variable_type='real',
    variable_boundaries=np.array([[0, 1]] * X_train.shape[1]),
    algorithm_parameters=algorithm_param,
)

model.run()

equal_weights = np.ones(X_train.shape[1])  # Set all weights to 1

# Compute predictions using equal weights
test_diff_baseline = np.abs(X_test[:, None] - X_train) * equal_weights
test_similarities_baseline = -np.sum(test_diff_baseline, axis=2)
test_neighbors_baseline = np.argpartition(test_similarities_baseline, -3, axis=1)[:, -3:]
test_predictions_baseline = np.mean(y_train[test_neighbors_baseline], axis=1)

# Compute MAE for baseline (equal weights)
baseline_mae = mean_absolute_error(y_test, test_predictions_baseline)
print(f"\nBaseline MAE with Equal Weights: {baseline_mae:.2f}")

ga_runtime = time.time() - start_time
print(f"Genetic Algorithm Runtime: {ga_runtime:.2f} seconds")
optimized_weights = np.abs(model.output_dict['variable'])

# Get test predictions using training data as cases
test_diff = np.abs(X_test[:, None] - X_train) * optimized_weights
test_similarities = -np.sum(test_diff, axis=2)
test_neighbors = np.argpartition(test_similarities, -3, axis=1)[:, -3:]
test_predictions = np.mean(y_train[test_neighbors], axis=1)

final_mae = mean_absolute_error(y_test, test_predictions)
print(f"\nTest MAE with optimized weights: {final_mae:.2f}")
print("Optimized Weights:")
for name, weight in zip(preprocessor.get_feature_names_out(), optimized_weights):
    print(f"{name}: {weight:.4f}")


X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

weights = torch.nn.Parameter(torch.tensor(optimized_weights, dtype=torch.float32, requires_grad=True))

# Define optimizer
optimizer = optim.Adam([weights], lr=0.01)


start_time = time.time()

# Training loop using softmax-weighted similarity
for epoch in range(50):
    optimizer.zero_grad()

    #weighted Manhattan distance
    weighted_diff = torch.abs(X_train_torch[:, None] - X_train_torch) * weights
    # Negative sum gives similarity (larger means more similar)
    similarities = -torch.sum(weighted_diff, dim=2)

    # Instead of hard top-k selection, use softmax to create differentiable weights
    soft_weights = torch.nn.functional.softmax(similarities, dim=1)
    # Weighted prediction: each case is predicted as the softmax-weighted sum of all y_train values
    predictions = torch.matmul(soft_weights, y_train_torch)

    loss = torch.mean(torch.abs(predictions - y_train_torch))

    # Backpropagate the loss
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


fine_tuning_runtime = time.time() - start_time
print(f"Gradient-Based Fine-Tuning Runtime: {fine_tuning_runtime:.2f} seconds")
# Extract the fine-tuned weights
fine_tuned_weights = weights.detach().numpy()
print("\nFine-Tuned Weights:", fine_tuned_weights)


# Extract feature names after transformation
feature_names = preprocessor.get_feature_names_out()

# Ensure that the number of weights matches the number of features
assert len(feature_names) == len(fine_tuned_weights), "Mismatch between features and weights!"

# Print feature names with their corresponding weights
print("\nOptimized Weights:")
for name, weight in zip(feature_names, fine_tuned_weights):
    print(f"{name}: {weight:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# introduce noise
def add_noise(X, noise_level=0.05):

    noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
    return np.clip(X + noise, 0, 1)  # Ensuring values stay within valid bounds

# Evaluate MAE under different noise levels
noise_levels = [0.01, 0.05, 0.1, 0.2]
mae_results = []

print("\nEvaluating Sensitivity to Noise...")

for noise_level in noise_levels:
    X_test_noisy = add_noise(X_test, noise_level)


    test_diff_noisy = np.abs(X_test_noisy[:, None] - X_train) * optimized_weights
    test_similarities_noisy = -np.sum(test_diff_noisy, axis=2)
    test_neighbors_noisy = np.argpartition(test_similarities_noisy, -3, axis=1)[:, -3:]
    test_predictions_noisy = np.mean(y_train[test_neighbors_noisy], axis=1)

    # Compute MAE with noisy test data
    mae_noisy = mean_absolute_error(y_test, test_predictions_noisy)
    mae_results.append(mae_noisy)

    print(f"MAE with {noise_level*100}% noise: {mae_noisy:.4f}")

#Visualizations

# Plot MAE under different noise conditions
plt.figure(figsize=(8, 5))
plt.plot([nl*100 for nl in noise_levels], mae_results, marker='o', linestyle='-', color='b')
plt.xlabel("Noise Level (%)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Impact of Noise on Model Performance")
plt.grid()
plt.savefig("noise_impact_plot.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(model.report, marker='o', linestyle='-', color='r')
plt.xlabel("Iteration")
plt.ylabel("Fitness (Mean Absolute Error)")
plt.title("Genetic Algorithm Convergence Curve")
plt.grid()

# Save the convergence plot as PNG
plt.savefig("GA_convergence_plot.png", dpi=300, bbox_inches='tight')
plt.show()

print("GA Convergence plot saved as 'GA_convergence_plot.png'")