# Large-Scale Optimal Transport and Mapping Estimation

This repository contains the implementation and evaluation of methods for large-scale optimal transport (OT) and mapping estimation, as explored in the project inspired by Seguy et al. [2018]. The project focuses on scalable OT techniques and their applications, such as learning transport plans and Monge maps, with a detailed comparison to traditional algorithms.

## Features
- **Stochastic Gradient-Based Dual Potentials Learning**: Implementation of a scalable approach to compute regularized OT plans.
- **Barycentric Projection**: Learning Monge maps for practical applications like domain adaptation and generative modeling.
- **Comparative Analysis**: Evaluation of the stochastic method against traditional Sinkhorn algorithms.
- **Reproducible Experiments**: Configurable settings for experiments on toy datasets, including Gaussian and discrete distributions.

## Table of Contents
- [Experiments](#experiments)
- [Results](#results)
- [References](#references)

## Experiments

All experiments are in the `src/test.ipynb` notebook.

The following experiments are implemented:
1. **Transport Plan Convergence**: Compare the stochastic gradient approach to the Sinkhorn algorithm on toy datasets.
2. **Barycentric Projection**: Evaluate the quality of learned Monge maps and their generalization ability.
3. **Convergence Speed**: Assess performance as a function of runtime and number of samples.

Detailed results are available in the `reports` directory.

## Results

### Highlights
- **Stochastic Dual Approach**: Achieved comparable results to the Sinkhorn algorithm in discrete settings, with improved scalability for larger datasets.
- **Challenges in Parametrization**: Neural network-based potentials demonstrated struggle to converge and numerical instability, highlighting areas for further optimization.

## References
- Vivien Seguy, Bharath Bhushan Damodaran, Rémi Flamary, Nicolas Courty, Antoine Rolet, and Mathieu Blondel. *Large-scale optimal transport and mapping estimation.* [arXiv:1711.02283](https://arxiv.org/abs/1711.02283).
- Gabriel Peyré and Marco Cuturi. *Computational Optimal Transport.* [arXiv:1803.00567](https://arxiv.org/abs/1803.00567).
