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
  One train set and multiple test sets relative to various covariate shift

- [ ] **Run Models**:  
  Train several different models end evaluate the performances on the different test sets

- [ ] **Conclusions and Possible Improvements**:  
  Whats the best model? What is the most robust? What are possible improvements in the training phase to have better predictions after the shift?

- [ ] **Analysis Extended to a Real Dataset**
  Possible extension of the study

## Preliminary Division of Work

- **Giovanni:** Synthetic data generation and GAM model  
- **Tommaso:** Random Forest model and result analysis  
- **Andrea:** XGBoost model and analysis  
- **Giacomo:** Adaptation methods for XGBoost (or other models) to improve prediction accuracy after data shift.

## Synthetic Datasets


## References

https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html

Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90: 227–244, 2000.

