# Neural Approximate Sufficient Statistics

-------------------------------------------------------------------------

Coming soon (really)!


## Dependencies
* Python 3
* Pytorch


## Summary statistic
at /nn
* Mean-as-statistics: s = E[theta|x] as statistic (traditional method, non-sufficient)
* Infomax statistics (proposed): s = argmax_S I(theta, S) (proposed method, near-sufficient)


## Likelihood-free algorithms
at /algorithms
* Sequential Monte Carlo ABC (SMC-ABC)
* Sequential Neural Posterior Estimate (SNPE)
* Sequential Neural Likelihood (SNL)
* Sequential Neural Ratio (SNR)

## Inference tasks
at /problems
* Ising model (non-analytical but could be approximated, graph data)
* Ornstein-Uhlenbeck process  (analytical, sequence data)
* Gaussian copula model (analytical, i.i.d data)
* M/G/1 queueing model (non analytical, i.i.d data)
