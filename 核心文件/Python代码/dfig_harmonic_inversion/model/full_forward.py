# -*- coding: utf-8 -*-
"""端到端可微正向模型 F(p): p -> y
p=[V_H, alpha, yaw(rad), pitch(rad)]
y=[f_sh, I_sh, P_ripple, Vdc_ripple]
"""
import jax
import jax.numpy as jnp
from model.aerodynamic import aerodynamic_torque
from model.mechanical import solve_shaft_ode
from model.electrical import electrical_output

def forward_model(p):
    """完整复合映射 F = elec ∘ mech ∘ aero"""
    Tm, wr0 = aerodynamic_torque(p)
    wg, dwg_amp = solve_shaft_ode(Tm, wr0)
    y_out = electrical_output(wg, dwg_amp, Tm)
    return y_out

# JAX向量化批量计算
batch_forward = jax.vmap(forward_model, in_axes=0)
# 自动雅可比函数（验证梯度）
jac_forward = jax.jacobian(forward_model)