# -*- coding: utf-8 -*-
"""双质量块扭振可微ODE模型，diffrax适配新版接口"""
import jax
import jax.numpy as jnp
import diffrax
from config.params import *

def shaft_ode(t, x, args):
    """
    x = [wr, wg, theta_sh]
    args = [Tm]
    ODE方程 论文附录B
    """
    Tm, = args
    wr, wg, theta_sh = x
    Tsh = K_sh * theta_sh + D_sh * (wr - wg)
    d_wr = (Tm - Tsh - D_r * wr) / J_r
    d_wg = (Tsh - D_g * wg) / J_g
    d_theta_sh = wr - wg
    return jnp.array([d_wr, d_wg, d_theta_sh])

def solve_shaft_ode(Tm0, w_r0):
    """给定稳态气动转矩，求解轴系稳态+振荡分量"""
    x0 = jnp.array([w_r0, w_r0, 0.0])
    term = diffrax.ODETerm(shaft_ode)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, T_SIM]))
    # 修复diffrax新版接口，使用diffeqsolve
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T_SIM,
        dt0=DT,
        y0=x0,
        args=(Tm0,),
        saveat=saveat
    )
    wr_end, wg_end, theta_end = sol.ys[-1]
    # 提取转速振荡幅值（简化）
    dwg_amp = jnp.abs(wg_end - w_r0)
    # 修复：变量改为wg_end，不存在wg变量
    return wg_end, dwg_amp