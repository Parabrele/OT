# Large Scale Optimal Transport and Mapping Estimation

https://arxiv.org/pdf/1711.02283

## Detailed summary

D1, D2 two distribution
X1, X2 two sets of samples drawn from D1, D2
Goal: find an optimal map from D1 to D2

Proposed method:
Two steps:
1. Learn an OT plan using a stochastic dual approach for regularized OT
2. Estimate a Monge map as a DNN learned by approximating the barycentric projection of the previously-obtained OT plan.
    -> Generalization to new samples outside the support of the input measure
    -> Genuinely learns a map from D1 to D2, not from X1 to X2

## Applications of OT

- Finding a map between two distributions:
    - Domain adaptation : aligning two distributions and allowing to transfer knowledge from one to the other (e.g. evaluating a model trained on one domain on the other)
    - Sampling : generating samples from an unknown distribution by sampling from a known one and applying the learned map
    - Generative models : GANs, VAEs, etc. : cf. point 2 above
- Rest of the intro from the paper.

## Step 1: Learning the OT plan

## Step 2: Learning the map

Barycentric projection of a plan $\pi$ solution of the OT problem w.r.t. $d :  \mathcal{Y} \times \mathcal{Y} \rightarrow R^+$ the cost function between points in Y :
    $\widetilde{\pi}(x) = \argmin_{z} \mathbb{E}_{Y \sim \pi(.|x)}[d(y, z)]$

- If $d(y, z) = ||y - z||_2^2$, is the squared Euclidean distance, then the barycentric projection is the mean of the points in Y:
    - $\widetilde{\pi}(x) = \mathbb{E}_{y \sim \pi(.|x)}[y]$
- If $c(x, y) = ||x - y||_2^2$ is the squared Euclidean cost, $\widetilde{\pi}$ is an optimal map

## Experiments

- $\epsilon$ : regularization parameter (for entropy regularization)

- /!\ experiments on semi discrete case ! "In the semi-discrete setting (i.e. one measure is discrete and the other is continuous), SGD on the semi-dual objective proposed by Genevay et al. (2016) also converges at a rate of O(1/√k), whereas we only know that Alg. 1 converges to a stationary point in this non-convex case."

- /!\ Barycentric projection : $\widetilde{\pi}_{\#}\mu$ is only approximately equal to $\nu$ (the target measure) : how close ? Use the Wasserstein distance as a metric.

- Toy distributions :
    - Gaussians
    - Mixture of Gaussians

- Compare OT plans (step 1 only) between different methods :
    - discrete barycentric projection of optimal plan (network simplex algorithm)
    - discrete barycentric projection of entropy regularized optimal plan (Sinkhorn)
    - step 1 + bar proj
    - step 1 + step 2

- Compare maps (step 2) between different methods :
    - baselines :
        - 1-NN
        - gaussian kernel of all points in X1, centerend on $\widetilde{x}$
        - Sinkhorn + barycentric projection
    - proposed method

- Setup :
    - Identity plan (X1 = X2)
    - Close to identity plan (X1 = X2 + eps)
    - Identity map (D1 = D2 but X1 != X2)
    - D1 = mixture, D2 = slightly translated and transformed mixture, but such that there is a 1 to 1 correspondence between the components : is the optimal transport plan able to find this correspondence ? (if ground truth is that mixture m1 and m2 are associated, are points very likely from m1 sent to a very likely point from m2 ?)

- Metrics :
    - Proximity between $\Pi_1$ and $\Pi_2$ (OT plans):
        - Frobenius norm : standard way to compare two matrices
        - L1 norm : ~ statistical distance (or total variation distance)
        - KL divergence : $\sum_{i,j} \Pi_1(i,j) \log(\Pi_1(i,j) / \Pi_2(i,j))$
        - Wasserstein distance between the two plans

- Questions :
    - Can we learn the OT ? (only interested in the training loss -> 0)
    - How good is the generalization to new samples ? (loss on X1^(test))
    - How many samples do we need to get a good generalization ? Is it somewhat proportional to the number of free parameters in the distributions ?
    - Better than Sinkhorn ? (sinkhorn is O(n^2) in the number of samples. Sinkhorn does not generalize : we compare the loss on the whole OT plan with the same training set. For the generalization, use barycentric projection of the OT plan to get a map from D1 to D2)
        - Time for similar performance
        - Performance for similar time
        - Number of samples needed for similar performance
        - Heatmap time x samples -> performance
        - Plot lines of equal performance for different methods : black lines in the heatmap, reported in graphs

- Visualisations :
    - Color points of X1 with the color of the corresponding arrival points in X2
    - Show generalization to new samples by doing the same coloring

- Generative models :
    - Compare to GANs, VAEs, diffusion models (state of the art generative models)
    - Compare to baseline methods (GMM)
    - Two settings : simple (Gaussian -> MNIST) and refined (GMM -> MNIST)
    - Metrics : FID, IS, etc.

## Links with the class :

In the class, we have seen Sinkhorn algorithm for regularized OT :
- Compare compute time for same performance
- Compare performance for same compute time
- Compare generalization to new samples outside X1, X2
- In class we have seen many methods to learn maps from X1 to X2, but not from D1 to D2 except in very specific cases like gaussians.