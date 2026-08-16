# -*- coding: utf-8 -*-
"""电气层：定子间谐波、有功脉动、直流脉动 可微公式"""
import jax
import jax.numpy as jnp
from config.params import *

def calc_sh_freq_amp(wg, hr, Urh):
    """论文2.3间谐波频率fsh、电流Ish"""
    fr = wg / (2 * jnp.pi)
    fsh = jnp.abs(hr * f1 + (f1 - fr))
    wsh = 2 * jnp.pi * fsh
    # 导纳计算
    Zs = Rs + 1j * wsh * Ls
    Zr = Rs / sigma_p + 1j * wsh * Lr
    Yrs = 1 / (Zs * Zr + (1j * wsh * Lm)**2) * 1j * wsh * Lm
    Ish = Urh * jnp.abs(Yrs)
    return fsh, Ish

def ripple_pe(dwg_amp, Te):
    """定子有功脉动幅值 2.4.1"""
    return Te * dwg_amp

def ripple_dc(Pripple, wsh):
    """直流母线电压脉动 2.4.2"""
    Vripple = jnp.abs(Pripple / (wsh * Cdc * Vdc0))
    return Vripple

def electrical_output(wg, dwg_amp, Tm):
    """电气层输出四元观测 [f_sh, I_sh, P_ripple, Vdc_ripple]"""
    hr = 1
    Urh = 120.0
    fsh, Ish = calc_sh_freq_amp(wg, hr, Urh) # 修复函数名
    Te = Tm * 0.98
    Prip = ripple_pe(dwg_amp, Te)
    Vrip = ripple_dc(Prip, fsh * 2 * jnp.pi)
    return jnp.array([fsh, Ish, Prip, Vrip])