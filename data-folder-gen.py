import numpy as np
import argparse
import os

from utils import simulate_sem, is_dag

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num', type=int, default=None,
                        help='number of graphs')
    parser.add_argument('--nodes', type=int, default=None,
                        help='number of nodes')
    parser.add_argument('--edges', type=int, default=None,
                        help='number of edges')
    parser.add_argument('--samples', type=int, default=None,
                        help='number of time samples')
    parser.add_argument('--graph', type=str, default='None',
                        help='graph type')
    parser.add_argument('--var', type=str, default=None,
                        help='variance type')
    parser.add_argument('--noise', type=str, default=None,
                        help='noise type')
    parser.add_argument('--max', type=float, default=None,
                        help='maximum edge weights')
    parser.add_argument('--min', type=float, default=None,
                        help='minimum edge weights')


    args = parser.parse_args()
    cwd = os.getcwd()
    noise_var = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    if args.var =='ev':
        for s in noise_var:

            name = args.noise + '-' + args.graph + '-' + args.var + '-var' + str(int(s)) + '-' + str(args.nodes) + '-' + str(args.edges)
            res_path = os.path.join(cwd, name)
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            for i in range(1, args.num+1):
                X, W_gt, sigma_gt = simulate_sem(n_nodes=args.nodes, n_samples=args.samples, edges=args.edges, graph_type=args.graph, edge_type='weighted', var_type=args.var, noise=args.noise, var=s, w_range=((-args.max, -args.min), (args.min, args.max)), seed=i)
                assert is_dag(W_gt)==True
                np.save(os.path.join(res_path, "data{}.npy".format(i)), X)
                np.save(os.path.join(res_path, "DAG{}.npy".format(i)), W_gt)

    elif args.var =='nv':
        name = args.noise + '-' + args.graph + '-' + args.var + '-' + str(args.nodes) + '-' + str(args.edges)
        res_path = os.path.join(cwd, name)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        for i in range(1, args.num+1):
            X, W_gt, var_gt = simulate_sem(n_nodes=args.nodes, n_samples=args.samples, edges=args.edges, graph_type=args.graph, edge_type='weighted', var_type=args.var, noise=args.noise, var=1.0, w_range=((-args.max, -args.min), (args.min, args.max)), seed=i)
            assert is_dag(W_gt)==True
            np.save(os.path.join(res_path, "data{}.npy".format(i)), X)
            np.save(os.path.join(res_path, "DAG{}.npy".format(i)), W_gt)
            np.save(os.path.join(res_path, "var{}.npy".format(i)), var_gt)
    else:
        raise ValueError('Variance type error!')
