import numpy as np
from scipy.stats import multivariate_normal, bernoulli
from itertools import combinations_with_replacement

#np.random.seed(0)

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
    """
    Generate a polynomial function of the specified degree from the input samples.
    
    Parameters
    ----------
    samples : np.ndarray
        Input samples to generate the polynomial from.
    degree : int
        Degree of the polynomial.
    coefficients : list, optional
        Coefficients of the polynomial terms. If not provided, random coefficients are generated.
        
    Returns
    -------
    np.ndarray
        Polynomial values for the input samples.
    list
        Coefficients of the polynomial terms.
    """
    
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
    
    """
    Generate a binary target variable from a polynomial function of the input samples.
    
    Parameters
    ----------
    sample : np.ndarray
        Input samples to generate the polynomial from.
    degree : int
        Degree of the polynomial.
    coefficients : list, optional
        Coefficients of the polynomial terms. If not provided, random coefficients are generated.
        
    Returns
    -------
    np.ndarray
        Binary target variable for the input samples.
    np.ndarray
        Polynomial values for the input samples.
    list
        Coefficients of the polynomial terms.
    """
    
    polynomial_values, coef = polynomial(sample, degree, coefficients)

    prob1 = sigmoid(polynomial_values)

    y = np.random.binomial(1, prob1)

    return y, polynomial_values, coef

from multiprocessing import Pool, cpu_count
import pandas as pd

class DataGenerator:
    """
    Class to generate synthetic datasets with mixtures of two distributions.
    """
    def __init__(self, mean_train, covariance_train, coef_train, df_train, mix_probs, N_SAMPLES, N_FEATURES, DEGREE):
        self.mix_probs = mix_probs.copy()
        self.mean_train = mean_train.copy()
        self.covariance_train = covariance_train.copy()
        self.coef_train = coef_train.copy()
        self.df_train = df_train.copy()
        self.N_SAMPLES = N_SAMPLES
        self.N_FEATURES = N_FEATURES
        self.DEGREE = DEGREE

    def process_experiment(self, i, cov_limit):
        """
        Process a single experiment to generate a dataset with a mixture of two distributions.
        """
        experiment_data = {}
        
        mean_shift = attributes_quantile(self.df_train, 0.05)
        covariance_shift = random_cov(self.N_FEATURES, -cov_limit, cov_limit)

        for mix_prob in self.mix_probs:
            # Generate the mixture sample
            sample_mix = build_mixture_sample(self.N_SAMPLES, self.mean_train, self.covariance_train, mean_shift, covariance_shift, mix_prob=mix_prob)

            # Create the dataframe
            df_mix = pd.DataFrame(sample_mix, columns=[f'X{i+1}' for i in range(self.N_FEATURES)])

            # Create the target variable
            y_mix, _, _ = build_poly_target(sample_mix, self.DEGREE, coefficients=self.coef_train)
            df_mix['Y'] = y_mix

            # Save the dataset
            experiment_data[mix_prob] = df_mix

        return i, experiment_data

    def parallelize_experiments(self, N_EXP, n_jobs = 1, cov_limit=0.5):
        """
        Parallelize the generation of multiple datasets with mixtures of two distributions
        """
        
        # Prepare the arguments for the experiments
        args = [(i, cov_limit) for i in range(N_EXP)]
        
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_jobs = min(n_jobs, N_EXP)

        # Parallelize the experiments
        with Pool(n_jobs) as pool:
            results = pool.starmap(self.process_experiment, args)

        # Collect the results in a dictionary
        experiment_df = {i: data for i, data in results}

        return experiment_df