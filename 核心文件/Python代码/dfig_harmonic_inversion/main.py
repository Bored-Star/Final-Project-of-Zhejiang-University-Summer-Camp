# -*- coding: utf-8 -*-
import time
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import LBFGS
from model.full_forward import forward_model
from dataset.data_generator import generate_dataset
from inversion.hybrid_opt import hybrid_invert
from inversion.loss_func import loss_fn
from utils.plot_tools import plot_error_bar
from config.params import TOTAL_SAMPLES

def exp1_obs_dim_compare():
    key = jax.random.PRNGKey(42)
    p_true, y_obs_all, _ = generate_dataset(key)
    err_2d = []
    err_4d = []
    for i in range(TOTAL_SAMPLES):
        y4 = y_obs_all[i]
        y2 = y4[:2]
        p4, _ = hybrid_invert(y4, grad_mode="jax")
        def loss_2d(p, y2_obs):
            y_pred = forward_model(p)
            return (y2_obs[0] - y_pred[0])**2 + (y2_obs[1] - y_pred[1])
        def fun_2d(p):
            return loss_2d(p, y2)
        def valgrad_2d(p):
            l = loss_2d(p, y2)
            g = jax.grad(lambda pp: loss_2d(pp, y2))(p)
            return l, g
        opt2 = LBFGS(fun=fun_2d, value_and_grad=valgrad_2d, maxiter=150)
        p2, _ = opt2.run(jnp.array([10, 0.2, 0, 0]))
        err2 = jnp.mean(jnp.abs((p2 - p_true[i]) / (p_true[i] + 1e-6)))
        err4 = jnp.mean(jnp.abs((p4 - p_true[i]) / (p_true[i] + 1e-6)))
        err_2d.append(float(err2))
        err_4d.append(float(err4))
    print(f"【二维观测方案】平均相对误差：{np.mean(err_2d):.4f}")
    print(f"【本文四维方案(JAX梯度)】平均相对误差：{np.mean(err_4d):.4f}")
    plot_error_bar(p_true, np.array(p4), "./output/figs/4d_vs_2d_error.png")

def exp_grad_compare():
    """JAX自动微分 vs 伴随梯度 耗时+精度对比"""
    key = jax.random.PRNGKey(999)
    p_true, y_obs_all, _ = generate_dataset(key)
    time_jax = []
    time_adj = []
    err_jax = []
    err_adj = []
    for i in range(TOTAL_SAMPLES):
        y4 = y_obs_all[i]
        # JAX梯度计时
        t0 = time.time()
        p4_jax, loss_jax = hybrid_invert(y4, grad_mode="jax")
        t_jax = time.time() - t0
        # 伴随梯度计时
        t0 = time.time()
        p4_adj, loss_adj = hybrid_invert(y4, grad_mode="adj")
        t_adj = time.time() - t0
        # 误差
        ej = jnp.mean(jnp.abs((p4_jax - p_true[i]) / (p_true[i] + 1e-6)))
        ea = jnp.mean(jnp.abs((p4_adj - p_true[i]) / (p_true[i])))
        time_jax.append(t_jax)
        time_adj.append(t_adj)
        err_jax.append(float(ej))
        err_adj.append(float(ea))
    print("========== 梯度方法对比 ==========")
    print(f"JAX自动微分：平均耗时 {np.mean(time_jax):.4f}s，平均误差 {np.mean(err_jax):.4f}")
    print(f"伴随状态梯度：平均耗时 {np.mean(time_adj):.4f}s，平均误差 {np.mean(err_adj):.4f}")

if __name__ == "__main__":
    print("========== 1、批量生成合成数据集 ==========")
    key_root = jax.random.PRNGKey(1234)
    p_t, y_o, y_t = generate_dataset(key_root)
    print("数据集生成完成")
    print("\n========== 2、观测维度消融实验 ==========")
    exp1_obs_dim_compare()
    print("\n========== 3、梯度对比实验(JAX/伴随) ==========")
    exp_grad_compare()
    print("全部实验运行完毕，图表已保存")