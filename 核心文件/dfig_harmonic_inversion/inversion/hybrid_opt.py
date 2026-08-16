# -*- coding: utf-8 -*-
"""贝叶斯全局搜索 + LBFGS局部精化，支持JAX/伴随两种梯度
"""
import numpy as np
import jax.numpy as jnp
from bayes_opt import BayesianOptimization
from jaxopt import LBFGS
from inversion.loss_func import loss_fn, grad_loss, loss_grad_adj_single

pbounds = {
    "vh": (4.0, 23.0),
    "alpha": (0.12, 0.38),
    "yaw": (-0.48, 0.48),
    "pitch": (0.03, 0.40)
}

def bayes_search(y_obs, init_iter=25):
    def black_box(vh, alpha, yaw, pitch):
        if not (3.0 <= vh <= 25.0
                and 0.1 <= alpha <= 0.4
                and -0.5236 <= yaw <= 0.5236
                and 0.0 <= pitch <= 0.4363):
            return -1e12
        p = jnp.array([vh, alpha, yaw, pitch])
        loss = loss_fn(p, y_obs)
        return -float(loss)
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
        allow_duplicate_points=True
    )
    optimizer.maximize(n_iter=init_iter)
    best = optimizer.max["params"]
    p_init = jnp.array([best["vh"], best["alpha"], best["pitch"], best["yaw"]])
    return p_init

def lbfgs_refine(p_init, y_obs, max_iter=150, grad_mode="jax"):
    def fun(p):
        return loss_fn(p, y_obs)
    def value_and_grad(p):
        loss_val = loss_fn(p, y_obs)
        if grad_mode == "jax":
            g = grad_loss(p, y_obs)
        elif grad_mode == "adj":
            g = loss_grad_adj_single(p, y_obs)
        else:
            raise ValueError("grad_mode仅支持 jax / adj")
        return loss_val, g
    solver = LBFGS(fun=fun, value_and_grad=value_and_grad, maxiter=max_iter)
    res = solver.run(p_init)
    p_opt = res.params
    loss_min = res.state.value
    return p_opt, loss_min

def hybrid_invert(y_obs, grad_mode="jax"):
    p_start = bayes_search(y_obs)
    # 修复：补充 y_obs 位置实参
    p_opt, loss_min = lbfgs_refine(p_start, y_obs, max_iter=150, grad_mode=grad_mode)
    return p_opt, loss_min