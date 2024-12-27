import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd



# Define the polynomial function (no interactions)
def polynomial_features(X):
    interaction_term = X[:, 0] * X[:, 1]  # Calcola il prodotto delle due feature
    return np.column_stack([np.ones(len(X)), X, X**2, interaction_term]) # bias, linear, quadratic and interaction

# Define the logistic (sigmoid) function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])


# Add a function to parameterize pi, covariance scale, and mean shift, and save datasets in a chosen folder:
def generate_datasets(pi=0.9, cov_scale=3.5, mean_shift_factor=1.5, output_folder="generated_data"):
    import os
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    # Build first distribution
    A = np.random.rand(2, 2) * 10
    cov0 = np.dot(A, A.transpose())
    mean0 = [10, 20]
    N = 1000
    X = np.random.multivariate_normal(mean0, cov0, N)

    # Polynomial features
    X_poly = polynomial_features(X)
    z = np.dot(X_poly, weights)
    probabilities = sigmoid(z)
    y = np.random.binomial(1, probabilities)
    df = pd.DataFrame(X, columns=['feature_1','feature_2'])
    df['target'] = y
    df.to_csv(os.path.join(output_folder, 'original.csv'), index=False)

    # Build second distribution
    from scipy.stats import ortho_group
    random_orthogonal_matrix = ortho_group.rvs(dim=2)
    cov1 = np.dot(random_orthogonal_matrix, np.dot(cov0, random_orthogonal_matrix.T))
    cov1 *= cov_scale
    mean1 = [x * mean_shift_factor for x in mean0]

    # Generate mixture
    samples = np.random.choice([0, 1], size=N, p=[pi, 1 - pi])
    X_mix = np.zeros((N, 2))
    for i in range(N):
        X_mix[i] = np.random.multivariate_normal(mean0 if samples[i]==0 else mean1,
                                                 cov0 if samples[i]==0 else cov1)
    # Second dataset polynomial features
    X_poly_mix = polynomial_features(X_mix)
    z_mix = np.dot(X_poly_mix, weights)
    probs_mix = sigmoid(z_mix)
    y_mix = np.random.binomial(1, probs_mix)
    df1 = pd.DataFrame(X_mix, columns=['feature_1','feature_2'])
    df1['target'] = y_mix
    df1.to_csv(os.path.join(output_folder, f'shifted_{str(pi).replace(".", "")}.csv'), index=False)
    
    
