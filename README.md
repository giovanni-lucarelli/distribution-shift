# Project Goal

This project aims to assess how simple *covariate shift* in the covariate distribution impact the performance of various robust models, for a synthetic binary  classification problem. Specifically, we evaluate the extent of performance degradation under different scenarios of covariate shift and possible strategies to overcome this problem (e.g. overfitting, weighted test?). 
1. toy problem: synthetic generated data
2. real dataset (hous price, iris?)


## Table of Contents
- [Project Goal](#project-goal)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Theoretical description](#theoretical-description)
  - [Generic Roadmap](#generic-roadmap)
  - [Preliminary Division of Work](#preliminary-division-of-work)
  - [References](#references)


## Datasets

All datasets consist of 3 features (X1, X2, X3), a target variable (Y) and a total of 1000 datapoints. The features are sampled from a multinormal distribution with randomly chosen mean and covariance matrix. The variable Y is computed by a polynomial of order 2 (with all possible interactions and randomly chosen coefficients) to which a sigmoid is applied to scale it in the range $[0,1]$. This last value obtained is then used as the probability of obtaining 1 in a Bernoulli trial.

The dataset `train.csv` contains the "original" distribution that will be used to train the models; `shift.csv` contains values ​​of the features $X_i$ sampled from a new multinormal distribution (centered in the 90th quartile of the original and with random covariance) and Y computed on the new sample as described above (with the same coefficients - *no concept shift!*).

The `mix_0.j.csv` dataset contains statistical mixtures of train and shift for different values ​​of mixture $0.j$, in particular $0.j$ is the probability of sampling from the shifted distribution.


## Theoretical description
Citing Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

The most basic form of dataset shift occurs when the data is generated according to a model $P(y|x)P(x)$ and where the distribution $P(x)$ changes between training and test scenarios. As only the covariate distribution changes, this has been called covariate shift [Shimodaira, 2000].
It will perhaps come as little surprise that the fact that the covariate distribution changes should have no effect on the model $P(y|x^∗)$. Intuitively this makes sense.

**Is there really no model implication?**

**Are some models more robust than others w.r.t. simple covariate shift?**


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
