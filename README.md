# ![colide](https://github.com/SAMiatto/colide/blob/main/logo/logo.png)

CoLiDE is a framework for learning linear directed acyclic graphs (DAGs) from observational data. Recognizing that DAG learning from observational data is in general an NP-hard problem, recent efforts have advocated a continuous relaxation approach which offers an efficient means of exploring the space of DAGs. We propose a new convex score function for sparsity-aware learning of linear DAGs, which incorporates concomitant estimation of scale parameters to enhance DAG topology inference using continuous first-order optimization. We augment this least-square-based score function with a smooth, nonconvex acyclicity penalty term to arrive at CoLiDE (**Co**ncomitant **Li**near **D**AG **E**stimation), a simple regression-based criterion that facilitates *efficient* computation of gradients and estimation of exogenous noise levels via closed-form expressions.


## Citation

This is an official implementation of the following paper:

S. S. Saboksayr, G. Mateos, and M. Tepper, [CoLiDE: Concomitant linear DAG estimation,][colide] [Proc. Int. Conf. Learn. Representations (ICLR)](https://iclr.cc/Conferences/2024), Vienna, Austria, May 7-11, 2024.

[colide]: https://arxiv.org/abs/2310.02895

If you find this code beneficial, kindly consider citing:

### BibTeX

```bibtex
@inproceedings{saboksayr2023colide,
  title={{CoLiDE: Concomitant Linear DAG Estimation}},
  author={Saboksayr, Seyed Saman and Mateos, Gonzalo and Tepper, Mariano},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Getting started

### Install the packages

We recommend using a virtual environment via `virtualenv` or `conda`, and use `pip` to install the requirements.

```bash
$ pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- `numpy`
- `scipy`
- `tqdm`
- `networkx`

### Running a simple demo

The simplest way to try out CoLiDE is by executing a straightforward example. 

```bash
$ python main.py --nodes 10 --edges 20 --samples 1000 --graph er --vartype ev --seed 0
```

This example initially generates a random Erdos-Renyi DAG with 10 nodes and 20 edges. Subsequently, utilizing a linear SEM, 1000 i.i.d samples are generated, assuming equal noise variance and a Gaussian noise distribution. Next, both CoLiDE-EV and CoLiDE-NV are applied to this data, and the graph recovery performance will be displayed.


## An Overview of CoLiDE

### Problem statement

Given the data matrix $\mathbf{X}$ adhering to a linear SEM, our goal is to learn the latent DAG $\mathcal{G} \in \mathbb{D}$ by estimating its adjacency matrix $\mathbf{W}$ as the solution to the optimization problem

$$ \min_{\mathbf{W}} \quad \mathcal{S} (\mathbf{W}) \quad \text{subject to} \quad \mathcal{G} (\mathbf{W}) \in \mathbb{D}, $$

where $\mathcal{S} (\mathbf{W})$ is a data-dependent score function to measure the quality of the candidate DAG. Irrespective of the criterion, the non-convexity comes from the combinatorial acyclicity constraint $\mathcal{G} (\mathbf{W}) \in \mathbb{D}$.

Noteworthy methods advocate an exact acyclicity characterization using nonconvex, smooth functions $\mathcal{H}:\mathbb{R}^{d \times d}\mapsto \mathbb{R}$ of the adjacency matrix, whose zero level set is $\mathbb{D}$. One can thus relax the combinatorial constraint $\mathcal{G}(\mathbf{W}) \in \mathbb{D}$ by instead enforcing $\mathcal{H}(\mathbf{W})=0$, and tackle the DAG learning problem using standard continuous optimization algorithms.

Minimizing $\mathcal{S}(\mathbf{W}) = \frac{1}{2n}|| \mathbf{X}  - \mathbf{W}^{\top} \mathbf{X} ||_{F}^2 + \lambda || \mathbf{W} ||_1$ subject to a smooth acyclicity constraint $\mathcal{H}(\mathbf{W})=0$: (i) requires carefully retuning $\lambda$ when the unknown SEM noise variance changes across problem instances; and (ii) implicitly relies on limiting homoscedasticity assumptions due to the ordinary LS loss. To address issues (i)-(ii), here we propose a new convex score function for linear DAG estimation that incorporates concomitant estimation of scale. This way, we obtain a procedure that is robust (both in terms of DAG estimation performance and parameter fine-tuning) to possibly heteroscedastic exogenous noise profiles.

### CoLiDE-EV

We start our exposition with a simple scenario whereby all exogenous variables $z_1,\ldots,z_d$ in the linear SEM have identical variance $\sigma^2$. Building on the [smoothed concomitant lasso](https://arxiv.org/abs/1606.02702), we formulate the problem of jointly estimating the DAG adjacency matrix $\mathbf{W}$ and the exogenous noise scale $\sigma$ as

$$
\min_{\mathbf{W}, \sigma \geq \sigma_0} \quad \underbrace{ \frac{1}{2 n \sigma} || \mathbf{X} - \mathbf{W}^{\top}\mathbf{X} ||_{F}^2 + \frac{d \sigma}{2} + \lambda || \mathbf{W} ||_1 }\_{:=\mathcal{S}(\mathbf{W},\sigma)} \quad \text{subject to} \quad \mathcal{H}(\mathbf{W})=0.
$$

Notably, the weighted, regularized LS score function $\mathcal{S}(\mathbf{W},\sigma)$ is now also a function of $\sigma$, and is jointly convex in $\mathbf{W}$ and  $\sigma$. Due to the rescaled residuals, $\lambda$ in $\mathcal{S}(\mathbf{W},\sigma)$ decouples from $\sigma$ as minimax optimality now requires $\lambda \asymp \sqrt{\log d / n}$. Of course, the optimization problem is still nonconvex by virtue of the acyclicity constraint $\mathcal{H}(\mathbf{W})=0$. 

With regards to the choice of the acyclicity function, we select $\mathcal{H}\_{\text{ldet}} (\mathbf{W}, s) = d \text{log}(s) - \text{log}(\text{det} (s \mathbf{I} - \mathbf{W} \circ \mathbf{W}))$ based on its more favorable gradient properties in addition to several other compelling reasons outlined in [DAGMA](https://arxiv.org/abs/2209.08037). Motivated by our choice of the acyclicity function, we solve the constrained optimization problem by solving a sequence of unconstrained problems where $\mathcal{H}_{\text{ldet}}$ is dualized and viewed as a regularizer. Given a decreasing sequence of values $\mu_k \to 0$, at step $k$ of the COLIDE-EV (equal variance) algorithm one solves

$$
\min_{\mathbf{W}, \sigma \geq \sigma_0} \quad \mu_k \left[ \frac{1}{2 n \sigma} \| \mathbf{X} - \mathbf{W}^{\top}\mathbf{X} \|_{F}^2 + \frac{d \sigma}{2} + \lambda \| \mathbf{W} \|_1 \right] + \mathcal{H}\_{\text{ldet}}(\mathbf{W}, s_k),
$$

where the schedule of hyperparameters $\mu_k\geq 0$ and $s_k>0$ must be prescribed prior to implementation. The additional constraint $\sigma \geq \sigma_0$ safeguards against potential ill-posed scenarios where the estimate $\hat{\sigma}$ approaches zero.

CoLiDE-EV jointly estimates the noise level $\sigma$ and the adjacency matrix $\mathbf{W}$ for each $\mu_k$. To this end, we rely on (inexact) block coordinate descent (BCD) iterations. This cyclic strategy involves fixing $\sigma$ to its most up-to-date value and minimizing the objective function inexactly w.r.t. $\mathbf{W}$, subsequently updating $\sigma$ in closed form given the latest $\mathbf{W}$ via

$$
\hat{\sigma} = \max\left(\sqrt{\text{Tr}\left( (\mathbf{I} - \mathbf{W})^{\top} \text{cov}(\mathbf{X}) (\mathbf{I} - \mathbf{W})\right)/d},\sigma_0\right),
$$

where $\text{cov}(\mathbf{X}) := \frac{1}{n} \mathbf{X} \mathbf{X}^{\top}$ is the precomputed sample covariance matrix. The mutually-reinforcing interplay between noise level and DAG estimation should be apparent. There are several ways to inexactly solve the $\mathbf{W}$ subproblem using first-order methods. Here, we run a single step of the ADAM optimizer to refine $\mathbf{W}$. We observed that running multiple ADAM iterations yields marginal gains, since we are anyways continuously re-updating $\mathbf{W}$ in the BCD loop. This process is repeated until either convergence is attained, or, a prescribed maximum iteration count is reached.

### CoLiDE-NV

We also address the more challenging endeavor of learning DAGs in heteroscedastic scenarios, where noise variables have non-equal variances (NV) $\sigma_1^2,\ldots,\sigma_d^2$. Bringing to bear ideas from the [generalized concomitant multi-task lasso](https://arxiv.org/abs/1705.09778) and mimicking the optimization approach for the EV case discussed earlier, we propose the CoLiDE-NV estimator

$$
\min_{\mathbf{W}, \boldsymbol{\Sigma} \geq \boldsymbol{\Sigma}_0} \quad \mu_k \left[ \frac{1}{2n} \text{Tr} \left( (\mathbf{X} - \mathbf{W}^{\top}\mathbf{X})^{\top} \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \mathbf{W}^{\top}\mathbf{X}) \right) + \frac{1}{2} \text{Tr}(\boldsymbol{\Sigma}) + \lambda \| \mathbf{W} \|_1 \right] + \mathcal{H}\_{\text{ldet}}(\mathbf{W}, s_k).
$$

Note that $\boldsymbol{\Sigma}=\text{diag}(\sigma_1,\ldots,\sigma_d)$ is a diagonal matrix of exogenous noise *standard deviations* (hence not a covariance matrix). A closed form solution for $\boldsymbol{\Sigma}$ given $\mathbf{W}$ is also readily obtained,

$$
\hat{\boldsymbol{\Sigma}} = \max\left(\sqrt{\text{diag}\left( (\mathbf{I} - \mathbf{W})^{\top} \text{cov}(\mathbf{X}) (\mathbf{I} - \mathbf{W}) \right)},\boldsymbol{\Sigma}_0\right),
$$

where $\sqrt{(\cdot)}$ is meant to be taken element-wise. 

## Acknowledgments

We express our gratitude to the authors of the [DAGMA repository][dagma-repository] for providing their code. A portion of our code is derived from their implementation, particularly incorporating their acyclicity function and optimization scheme.

[dagma-repository]: https://github.com/kevinsbello/dagma
