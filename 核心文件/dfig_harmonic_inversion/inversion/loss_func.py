# -*- coding: utf-8 -*-
"""损失函数 + JAX原生梯度 + 伴随梯度对外接口
"""
import jax
import jax.numpy as jnp
from model.full_forward import forward_model
from config.params import *
# 修正导入：导入正确函数 adjoint_solve_single
from inversion.adjoint_grad import adjoint_solve_single

def loss_fn(p, y_obs, w_f=1.0, w_p=1.0, w_v=1.0, gamma=0.01):
    """带Tikhonov正则加权最小二乘损失"""
    y_pred = forward_model(p)
    f_err = (y_obs[0] - y_pred[0]) ** 2 * w_f
    i_err = (y_obs[1] - y_pred[1])
    p_err = (y_obs[2] - y_pred[2]) ** 2 * w_p
    v_err = (y_obs[3] - y_pred[3]) ** 2 * w_v
    residual = f_err + i_err + p_err + v_err
    p0 = jnp.array([(V_H_RANGE.sum())/2, (ALPHA_RANGE.sum())/2, (YAW_RANGE.sum())/2, (PITCH_RANGE.sum())/2])
    reg = gamma * jnp.sum((p - p0)**2)
    total_loss = residual + reg
    return total_loss

# JAX自动微分梯度（原生）
grad_loss = jax.grad(loss_fn)
batch_loss = jax.vmap(loss_fn, in_axes=(0, 0, None, None, None, None))

# 对外单样本伴随梯度接口（给hybrid_opt调用）
def loss_grad_adj_single(p, y_obs):
    return adjoint_solve_single(p, y_obs, loss_fn)