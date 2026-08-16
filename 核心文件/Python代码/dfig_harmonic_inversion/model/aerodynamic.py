# -*- coding: utf-8 -*-
"""气动层可微模型：风切变三阶泰勒+塔影平滑修正"""
import jax
import jax.numpy as jnp
from config.params import *
from utils.math_helper import tanh_smooth_mask

def wind_shear_term(r, theta, alpha):
    """三阶泰勒展开风切变扰动 Ws(r,θ)"""
    term1 = alpha * (r / H) * jnp.cos(theta)
    term2 = alpha * (alpha - 1) / 2 * (r / H) ** 2 * jnp.cos(theta) ** 2
    term3 = alpha * (alpha - 1) * (alpha - 2) / 6 * (r / H) ** 3 * jnp.cos(theta) ** 3
    return term1 + term2 + term3

def m_coeff(alpha):
    """V0/VH 修正系数 m"""
    ratio = (R / H) ** 2
    return 1 + alpha * (alpha - 1) * ratio / 8

def tower_shadow_single(r, theta, alpha):
    """单叶片塔影扰动，tanh平滑限制θ∈[π/2, 3π/2]"""
    m = m_coeff(alpha)
    mask = tanh_smooth_mask(theta, jnp.pi/2, 3*jnp.pi, k=25)
    numer = m * (r**2 * jnp.sin(theta)**2 - x_blade_tower**2)
    denom = (r**2 * jnp.sin(theta)**2 + x_blade_tower**2) ** 2
    v_ts = a_tower**2 * numer / denom
    return v_ts * mask

def calc_v_eq(theta_list, alpha, V_H):
    """等效风速 Veq 三叶片求和"""
    theta_b = jnp.array([theta_list[0], theta_list[0]+2*jnp.pi/3, theta_list[0]+4*jnp.pi/3])
    ws_sum = 0.0
    ts_sum = 0.0
    for th in theta_b:
        # 修复：补充 alpha 入参
        ws_sum += wind_shear_term(R, th, alpha)
        ts_sum += tower_shadow_single(R, th, alpha)
    ws_comp = V_H * alpha * (alpha - 1) * (R/H)**3 / 60 * jnp.cos(3 * theta_list[0])
    v_eq = V_H * (1 + ws_sum/3 + ws_comp + ts_sum/3)
    return v_eq

def cp_func(lam, beta):
    """简化可微Cp曲面，无分段，全程光滑"""
    c1 = 0.5176
    c2 = 116
    c3 = 0.4
    c4 = 5
    c5 = 21
    c6 = 0.0068
    lam_i = 1 / (lam + 0.08 * beta) - 0.035 / (beta**3 + 1)
    cp = c1 * (c2/lam_i - c3*beta - c4) * jnp.exp(-c5/lam_i) + c6 * lam
    return jnp.clip(cp, 0.0, 0.59)

def aerodynamic_torque(p):
    """输入参数 p = [V_H, alpha, yaw, pitch]
    输出气动转矩 Tm
    """
    V_H, alpha, yaw, beta = p
    theta0 = yaw  # 修复：原y未定义，替换为yaw
    v_eq = calc_v_eq([theta0], alpha, V_H)
    w_r_ref = jnp.clip(v_eq * R / 8, 0.5, 2.2)
    lam = w_r_ref / (v_eq + 1e-8)
    cp = cp_func(lam, beta)
    Pm = 0.5 * rho * jnp.pi * R**2 * (v_eq ** 3) * cp
    Tm = Pm / (w_r_ref + 1e-8)
    return Tm, w_r_ref