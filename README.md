# Distribution Shift

This project aims to assess how simple *covariate shift* in the covariate distribution impact the performance of 
various robust models, for a synthetic binary  classification problem. Specifically, we evaluate the extent of 
performance degradation under different scenarios of covariate shift and possible strategies to overcome this problem.

Questions:

- **How does a model's performance degrade following a simple distribution shift?**
- **Are some models more robust than others w.r.t. simple covariate shift?**
- **How can we improve performance after the shift?**


## Table of Contents
- [Distribution Shift](#distribution-shift)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Generic Roadmap](#generic-roadmap)
  - [Preliminary Division of Work](#preliminary-division-of-work)
  - [References](#references)

## Datasets

All datasets consist of 3 features (X1, X2, X3), a target variable (Y) and a total of 1000 datapoints. The features are sampled from a multinormal distribution with randomly chosen mean and covariance matrix. The variable Y is computed by a polynomial of order 2 (with all possible interactions and randomly chosen coefficients) to which a sigmoid is applied to scale it in the range $[0,1]$. This last value obtained is then used as the probability of obtaining 1 in a Bernoulli trial.

The dataset `train.csv` contains the "original" distribution that will be used to train the models; `shift.csv` contains values ​​of the features $X_i$ sampled from a new multinormal distribution (centered in the 90th quartile of the original and with random covariance) and Y computed on the new sample as described above (with the same coefficients - *no concept shift!*).

The `mix_j.csv` dataset contains statistical mixtures of train and shift for different values ​​of mixture $j$, in particular $j$ is the probability of sampling from the shifted distribution.

In the `/data` folder there is also `Parameters.txt` that stores all the parameters used during the data generation, coulld be usefull for further analysis.

## Generic Roadmap

1. Data generation for a synthetic bynary classification problem
2. Fit of different models and study of performance degradation for various mixtures
3. Study on possible improvements

## Preliminary Division of Work

- **Giovanni:** Synthetic data generation and GAM model  
- **Tommaso:** Random Forest  
- **Andrea:** Boosting models   
- **Giacomo:** Adaptation methods to improve performance after data shift.

## References

https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html

Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90: 227–244, 2000.
