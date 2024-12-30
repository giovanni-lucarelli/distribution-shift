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
