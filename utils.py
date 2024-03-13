import numpy as np
import networkx as nx
import time

def count_accuracy(B_bin_true, B_bin_est, check_input=False):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def simulate_sem(n_nodes, n_samples, edges, graph_type='er', edge_type='weighted', var_type='ev', noise='normal', var=1.0, w_range=((-2.0, -0.5), (0.5, 2.0)), seed=123):

    rng = np.random.default_rng(seed=seed)
    if graph_type == 'er':
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        adj = nx.to_numpy_array(G)
        U_mask = np.triu(adj, k=1)
        P = np.eye(n_nodes)
        P = P[:, rng.permutation(n_nodes)]
        W = P @ U_mask @ P.T
    elif graph_type == 'sf':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m, seed=seed)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)
    else:
        raise ValueError('Unknown graph type')
        
    assert nx.is_weighted(G)==False
    assert nx.is_empty(G)==False

    if edge_type == 'binary':
        W_weighted = W.copy()
    elif edge_type == 'weighted':
        W_weighted = np.zeros(W.shape)
        S = np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights
    else:
        raise ValueError('Unknown edge type')
    G_sem = nx.DiGraph(W_weighted)

    X = np.zeros((n_samples, n_nodes))
    ordered_vertices = list(nx.topological_sort(G_sem))
    assert len(ordered_vertices) == n_nodes
    var_nv = rng.uniform(0.5,10.0,n_nodes)
    
    t_start = time.time()
    for j in ordered_vertices:
        parents = list(G_sem.predecessors(j))
        eta = X[:, parents].dot(W_weighted[parents, j])
        if var_type =='ev':
            if noise == 'normal':
                scale = np.sqrt(var)
                X[:, j] = eta + rng.normal(scale=scale, size=(n_samples))
            elif noise == 'exp':
                scale = np.sqrt(var)
                X[:, j] = eta + rng.exponential(scale=scale, size=(n_samples))
            elif noise == 'laplace':
                scale = np.sqrt(var / 2.0)
                X[:, j] = eta + rng.laplace(loc=0.0, scale=scale, size=(n_samples))
            elif noise == 'gumbel':
                scale = np.sqrt(6.0 * var) / np.pi
                X[:, j] = eta + rng.gumbel(loc=0.0, scale=scale, size=(n_samples))
            else:
                raise ValueError('Noise type error!')
        elif var_type =='nv':
            if noise == 'normal':
                scale = np.sqrt(var_nv[j])
                X[:, j] = eta + rng.normal(scale=scale, size=(n_samples))
            elif noise == 'exp':
                scale = np.sqrt(var_nv[j])
                X[:, j] = eta + rng.exponential(scale=scale, size=(n_samples))
            elif noise == 'laplace':
                scale = np.sqrt(var_nv[j] / 2.0)
                X[:, j] = eta + rng.laplace(loc=0.0, scale=scale, size=(n_samples))
            elif noise == 'gumbel':
                scale = np.sqrt(6.0 * var_nv[j]) / np.pi
                X[:, j] = eta + rng.gumbel(loc=0.0, scale=scale, size=(n_samples))
            else:
                raise ValueError('Noise type error!')
        else:
            raise ValueError('Variance type error!')

    t_end = time.time()
    assert is_dag(W_weighted)==True
    print('The data generation is finished! It took', t_end-t_start, 'seconds.')
    
    return X, W_weighted, var_nv

def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def to_dag(W, thr=0.3):
    A = np.copy(W)
    A[np.abs(A) <= thr] = 0

    if is_dag(A):
        return A
    
    nonzero_indices = np.where(A != 0)
    weight_indices_ls = list(zip(A[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))
    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(A):
            break
        A[j, i] = 0

    return A