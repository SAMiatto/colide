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
