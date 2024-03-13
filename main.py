import os
import argparse
import numpy as np
import time
from utils import simulate_sem, count_accuracy, to_dag
from model import colide_ev, colide_nv

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    parser.add_argument('--nodes', type=int, default=10,
                        help='number of nodes')
    parser.add_argument('--edges', type=int, default=20,
                        help='number of edges')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of time samples')
    parser.add_argument('--graph', type=str, default='er',
                        help='graph type')
    parser.add_argument('--vartype', type=str, default='ev',
                        help='variance type')
    parser.add_argument('--var', type=float, default=1.0,
                        help='noise variance')
    parser.add_argument('--noise', type=str, default='normal',
                        help='noise type')
    parser.add_argument('--max', type=float, default=2.0,
                        help='maximum edge weights')
    parser.add_argument('--min', type=float, default=0.5,
                        help='minimum edge weights')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed number')

    args = parser.parse_args()

    #####################
    # Generating signal #
    #####################

    X, W_gt, sigma_gt = simulate_sem(n_nodes=args.nodes, n_samples=args.samples, edges=args.edges, graph_type=args.graph, edge_type='weighted', var_type=args.vartype, noise=args.noise, var=args.var, w_range=((-args.max, -args.min), (args.min, args.max)), seed=args.seed)

    #####################
    # Running CoLiDE-EV #
    #####################

    model1 = colide_ev(seed=args.seed)
    t_start = time.time()
    W_hat_ev, sigma_est_ev = model1.fit(X, lambda1=0.05, T=4, s=[1.0, .9, .8, .7], warm_iter=2e4, max_iter=7e4, lr=0.0003)
    t_end = time.time()
    print(f'convergence time for CoLiDE-EV: {t_end-t_start:.4f}s')
    W_hat_post_ev = to_dag(W_hat_ev, thr=0.3)
    fdr_ev, tpr_ev, fpr_ev, shd_ev, pred_size_ev = count_accuracy(W_gt!=0, W_hat_post_ev!=0)

    #####################
    # Running CoLiDE-NV #
    #####################

    model2 = colide_nv(seed=args.seed)
    t_start = time.time()
    W_hat_nv, Sigma_est_nv = model2.fit(X, lambda1=0.05, T=4, s=[1.0, .9, .8, .7], warm_iter=2e4, max_iter=7e4, lr=0.0003)
    t_end = time.time()
    print(f'convergence time for CoLiDE-NV: {t_end-t_start:.4f}s')
    W_hat_post_nv = to_dag(W_hat_nv, thr=0.3)
    fdr_nv, tpr_nv, fpr_nv, shd_nv, pred_size_nv = count_accuracy(W_gt!=0, W_hat_post_nv!=0)

    ######################
    # Displaying Results #
    ######################

    print('=== CoLiDE-EV Results ===')
    print('SHD:', shd_ev, 'FDR:', fdr_ev, 'TPR:', tpr_ev, 'NNZ:', pred_size_ev)

    print('=== CoLiDE-NV Results ===')
    print('SHD:', shd_nv, 'FDR:', fdr_nv, 'TPR:', tpr_nv, 'NNZ:', pred_size_nv)