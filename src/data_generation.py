import numpy as np
from scipy.stats import multivariate_normal, bernoulli
from itertools import combinations_with_replacement

# Attribute generation functions

def attributes_quantile(df, probability):
    quantiles= df.quantile(probability)
    quantiles = quantiles[:-2] # Remove the last two elements
    return quantiles

def random_cov(num_features, low=-1, high=1):
  # Define the covariance matrix
  covariance = np.random.uniform(low, high, size=(num_features, num_features))
  covariance = np.dot(covariance, covariance.transpose()) # positive semidef

  return covariance

def build_multivariate_sample(num_samples, mean, covariance):
  # Create the multivariate normal distribution
  mvn = multivariate_normal(mean=mean, cov=covariance)

  # Generate random samples
  samples = mvn.rvs(size=num_samples)

  return samples

def build_mixture_sample(num_samples, mean1, cov1, mean2, cov2, mix_prob):
    """
    Build a random sample from a statistical mixture of two multivariate distributions.

    Parameters:
    - num_samples: Number of samples to generate.
    - mean1, cov1: Mean and covariance of the first multivariate normal distribution.
    - mean2, cov2: Mean and covariance of the second multivariate normal distribution.
    - mix_prob: Mixing probability for the first distribution (0 <= mix_prob <= 1).

    Returns:
    - samples: An array of generated samples.
    """
    # Create the two multivariate normal distributions
    mvn1 = multivariate_normal(mean=mean1, cov=cov1)
    mvn2 = multivariate_normal(mean=mean2, cov=cov2)

    # Generate Bernoulli trials to decide which distribution to sample from
    mixture_selection = bernoulli.rvs(mix_prob, size=num_samples)

    # Allocate space for the samples
    samples = np.zeros((num_samples, len(mean1)))

    # Generate samples based on the Bernoulli outcomes
    for i in range(num_samples):
        if mixture_selection[i] == 0:
            samples[i] = mvn1.rvs()
        else:
            samples[i] = mvn2.rvs()

    return samples

# Target generation functions

def polynomial(samples, degree, coefficients=None):
    
    #check if samples is 1-dim

    if len(samples.shape) < 2:
        print("Reshaping samples to 2D")
        samples = samples.reshape(-1, 1)
    
    num_samples = samples.shape[0]
    num_features = samples.shape[1]

    terms = []
    generated_coefficients = []  # To store generated coefficients if not provided

    # Generate all combinations of feature indices for terms up to the specified degree
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(num_features), d):
            terms.append(combo)

    if coefficients is None:
        # Generate random coefficients if not provided
        coefficients = [np.random.uniform(-1, 1) for _ in terms]
        generated_coefficients = coefficients

    polynomial_values = np.zeros(num_samples)
    for combo, coefficient in zip(terms, coefficients):
        # Compute the term for each sample
        term_values = np.prod(samples[:, combo], axis=1) if combo else np.ones(num_samples)
        polynomial_values += coefficient * term_values

    # Return the polynomial values and either the passed or generated coefficients
    return polynomial_values, (coefficients if generated_coefficients == [] else generated_coefficients)

def sigmoid(x):
    """
    Computes the sigmoid activation function element-wise.
    
    Parameters
    ----------
    x : np.ndarray
        Input array.
    
    Returns
    -------
    np.ndarray
        Sigmoid output, same shape as input.
    """
    return 1 / (1 + np.exp(-x))

def build_poly_target(sample, degree, coefficients=None):
  polynomial_values, coef = polynomial(sample, degree, coefficients)

  prob1 = sigmoid(polynomial_values)

  y = np.random.binomial(1, prob1)

  return y, polynomial_values, coef
