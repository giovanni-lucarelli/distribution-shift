# Distribution Shift

This project aims to assess how simple *covariate shift* in the covariate distribution impact the performance of 
various robust models, for a synthetic binary  classification problem. Specifically, we evaluate the extent of 
performance degradation under different scenarios of covariate shift and possible strategies to overcome this problem.

Questions:

- **How does a model's performance degrade following a simple distribution shift?**
- **Are some models more robust than others w.r.t. simple covariate shift?**
- **How can we improve performance after the shift?**

> [!note]
> [Here](<https://raw.githubusercontent.com/giovanni-lucarelli/distribution-shift/main/report/src/main.pdf>) you can find a more complete pdf version of the project report

## Table of Contents
- [Distribution Shift](#distribution-shift)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [Generic Roadmap](#generic-roadmap)
  - [Preliminary Division of Work](#preliminary-division-of-work)
  - [References](#references)

## Generic Roadmap

1. Data generation for a synthetic bynary classification problem
2. Fit of different models and study of performance degradation for various mixtures
3. Study on possible improvements

## References

https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html

Dataset Shift in Machine Learning. United Kingdom: MIT Press, 2009.

H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90: 227–244, 2000.
