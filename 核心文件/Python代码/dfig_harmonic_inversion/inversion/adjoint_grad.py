# -*- coding: utf-8 -*-
"""伴随状态梯度完整修复：维度匹配、时序降序、全状态梯度、积分修正"""
import jax
import jax.numpy as jnp
from diffrax import ODETerm, Dopri5, SaveAt, diffeqsolve

# 补齐全部模块导入
from model.aerodynamic import aerodynamic_torque
from model.mechanical import shaft_ode
from model.electrical import electrical_output
from model.full_forward import forward_model
from config.params import *

def shaft_ode_jac_x(t, x, args):
    Tm, = args
    wr, wg, th_sh = x
    df_dx = jnp.array([
        [-(D_r + D_sh)/J_r, D_sh/J_r, -K_sh/J_r],
        [D_sh/J_g, -(D_g + D_sh)/J_g, K_sh/J_g],
        [1.0, -1.0, 0.0]
    ])
    return df_dx

# 状态对扭矩Tm雅可比，仅输出3×1，不参与多参数
def shaft_ode_jac_p(t, x, args):
    Tm, = args
    return jnp.array([[1/J_r], [0.0], [0.0]])

def adjoint_solve_single(p, y_obs, loss_fn):
    y_pred = forward_model(p)
    loss_val = loss_fn(p, y_obs)
    dL_dy = jax.grad(lambda y: loss_fn(p, y))(y_pred)
    T, wr0 = aerodynamic_torque(p)
    x0 = jnp.array([wr0, wr0, 0.0])
    term_f = ODETerm(shaft_ode)
    solver = Dopri5()

    # 正向升序时间
    ts_forward = jnp.linspace(0.0, T_SIM, 200)
    saveat_forward = SaveAt(ts=ts_forward)
    sol_forward = diffeqsolve(
        term_f,
        solver,
        t0=0.0,
        t1=T_SIM,
        dt0=DT,
        y0=x0,
        args=(T,),
        saveat=saveat_forward
    )
    x_end = sol_forward.ys[-1]
    wr_end, wg_end, theta_end = x_end

    # 局部映射：状态→观测
    def output_at(x_in):
        wr_t, wg_t, th_t = x_in
        return electrical_output(wr_t, wg_t, T)

    # 完整三维终端伴随初值（修复之前仅wg单分量）
    dL_dx_end = jax.grad(lambda xx: loss_fn(p, output_at(xx)))(x_end)
    lambda_T = -dL_dx_end

    # 插值任意时刻系统状态
    def get_x_at_time(t):
        diff = jnp.abs(ts_forward - t)
        idx = jnp.argmin(diff)
        return sol_forward.ys[idx]

    # 伴随ODE右端
    def adjoint_ode(t, lam, args_adj):
        Tm_t = args_adj
        x_t = get_x_at_time(t)
        jac_x = shaft_ode_jac_x(t, x_t, (Tm_t,))
        A_T = jac_x.T
        dLdx_t = jax.grad(lambda xx: loss_fn(p, output_at(xx)))(x_t)
        return -A_T @ lam

    term_adj = ODETerm(adjoint_ode)
    # 反向使用降序ts，满足diffrax时序校验
    ts_back = jnp.flip(ts_forward)
    saveat_back = SaveAt(ts=ts_back)
    sol_adj = diffeqsolve(
        term_adj,
        solver,
        t0=T_SIM,
        t1=0.,
        dt0=-DT,
        y0=lambda_T,
        args=T,
        saveat=saveat_back
    )

    # 数值积分计算标量积分项 ∫λ^T ∂f/∂Tm dt
    lam_traj = sol_adj.ys
    dt_step = ts_forward[1] - ts_forward[0]
    integral_sum = 0.0
    for i in range(len(ts_forward)):
        t = ts_forward[i]
        x_t = sol_forward.ys[i]
        lam_t = lam_traj[i]
        jac_p = shaft_ode_jac_p(t, x_t, (T,))
        # lam_t(3,) @ jac_p(3,1) = 标量
        integral_sum += (lam_t @ jac_p)[0] * dt_step

    # 气动扭矩对4维参数p的梯度 (4,)
    dT_dp = jax.grad(lambda pp: aerodynamic_torque(pp)[0])(p)
    # 修复：标量 * 向量，不是矩阵@
    dL_dp = integral_sum * dT_dp
    return dL_dp

# 批量vmap
adj_loss_batch = jax.vmap(adjoint_solve_single, in_axes=(0, 0, None))