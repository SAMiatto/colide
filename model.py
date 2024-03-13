import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm


#===================================#
#   Equal variance  CoLiDE-EV       #
#===================================#

class colide_ev:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
            
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        loss = ((0.5 * np.trace(dif.T @ rhs)) / sigma) + (0.5 * sigma * self.d)
        G_loss = -rhs / sigma
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    dif = self.Id - W 
                    rhs = self.cov @ dif
                    sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            
            G_score = -mu * self.cov @ (self.Id - W) / sigma
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W 
            rhs = self.cov @ dif
            sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, sigma, True
    
    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
            warm_iter=3e4, max_iter=6e4, lr=0.0003, 
            checkpoint=1000, beta_1=0.99, beta_2=0.999,
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        self.sig_est = np.min(np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est
    
#===================================#
#   Non-equal variance  CoLiDE-NV   #
#===================================#

class colide_nv:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
            
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0/(sigma))
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / (2.0)
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    dif = self.Id - W
                    rhs = self.cov @ dif
                    sigma = np.sqrt(np.diag(dif.T @ rhs))
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            
            inv_SigMa = np.diag(1.0/(sigma))
            G_score = -mu * (self.cov @ (self.Id - W) @ inv_SigMa)
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W
            rhs = self.cov @ dif
            sigma = np.sqrt(np.diag(dif.T @ rhs))
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, sigma, True
    
    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
            warm_iter=3e4, max_iter=6e4, lr=0.0003, 
            checkpoint=1000, beta_1=0.99, beta_2=0.999, w_init=None,
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)
        if w_init is None:    
            self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
            self.sig_est = (np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        else:
            self.W_est = np.copy(w_init).astype(self.dtype)
            self.sig_est = (np.linalg.norm(self.X @ (self.Id - w_init), axis=0) / np.sqrt(self.n)).astype(self.dtype)

        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor
                
        return self.W_est, self.sig_est