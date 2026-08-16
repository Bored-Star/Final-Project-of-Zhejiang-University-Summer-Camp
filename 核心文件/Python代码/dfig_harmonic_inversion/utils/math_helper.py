# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

def tanh_smooth_mask(theta, theta_low, theta_high, k=20.0):
    """tanh平滑阶跃，替代塔影角度硬边界
    论文4.6节平滑方法
    """
    mask_low = 1.0 / (1.0 + jnp.exp(-k * (theta - theta_low)))
    mask_high = 1.0 / (1.0 + jnp.exp(k * (theta - theta_high)))
    return mask_low * mask_high

def rad2deg(x):
    return x * 180 / jnp.pi

def deg2rad(x):
    return x * jnp.pi / 180