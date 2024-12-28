# Project Goal

This project aims to assess how simple *covariate shift* in the covariate distribution impact the performance of various robust models, for a regression problem and for a (binary) classification problem. Specifically, we evaluate the extent of performance degradation under different scenarios of covariate shift and possible strategies to overcome this problem (e.g. overfitting, weighted test?). 
1. toy problem: syntetic generated data
2. real dataset (hous price, iris?)

## Theoretical description
Citing Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

The most basic form of dataset shift occurs when the data is generated according to a model $P(y|x)P(x)$ and where the distribution $P(x)$ changes between training and test scenarios. As only the covariate distribution changes, this has been called covariate shift [Shimodaira, 2000].
It will perhaps come as little surprise that the fact that the covariate distribution changes should have no effect on the model P(y|x∗). Intuitively this makes sense.

**Is there really no model implication?**

**Are some models more robust than others w.r.t. simple covariate shift?**


## Generic TODO

- [ ] **Generate a Simple Dataset**:  
  Create a dataset drawn from a multivariate normal distribution.

- [ ] **Generate Target Variables**:  
  Produce target variables using a logistic regression model with predictors defined by a linear combination of smooth functions, including an irreducible error term.

- [ ] **Create New Datasets**:  
  Generate a new multivariate dataset and a mixture dataset that combines the original dataset with the newly created one.

- [ ] **Run Models**:  
  Train various models on both the original dataset and the mixture dataset, exploring different mixture ratios.

- [ ] **Evaluate Performance**:  
  Assess model performance and analyze how it degrades as the distribution shifts.


## Preliminary Division of Work

- **Giovanni:** Synthetic data generation and GAM model  
- **Tommaso:** Random Forest model and result analysis  
- **Andrea:** XGBoost model and analysis  
- **Giacomo:** Adaptation methods for XGBoost to improve prediction accuracy after data shift (or, more generally, boosting methods applied to all three previous models to enhance their predictive power)

## Datasets

## References

https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html

Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90: 227–244, 2000.

