# Project Goal

This project aims to assess how distribution shifts in the covariate distribution impact the performance of various models. Specifically, we evaluate the extent of performance degradation under different scenarios of covariate shift.

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


**Preliminary Division of Work:**

- **Giovanni:** Synthetic data generation and GAM model  
- **Tommaso:** Random Forest model and result analysis  
- **Andrea:** XGBoost model and analysis  
- **Giacomo:** Adaptation methods for XGBoost to improve prediction accuracy after data shift (or, more generally, boosting methods applied to all three previous models to enhance their predictive power)
