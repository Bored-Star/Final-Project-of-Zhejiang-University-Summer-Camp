# -*- coding: utf-8 -*-
"""批量生成合成真值参数+带噪声观测数据"""
import jax
import jax.numpy as jnp
import numpy as np
from config.params import *
from model.full_forward import batch_forward

def generate_dataset(key, sample_num=TOTAL_SAMPLES):
    # 修复：原代码未给key赋值，删除无意义单独创建PRNGKey语句
    # 随机采样p真值
    key1, key2, key3, key4 = jax.random.split(key, 4)
    V_H = jax.random.uniform(key1, (sample_num,), minval=V_H_RANGE[0], maxval=V_H_RANGE[1])
    alpha = jax.random.uniform(key2, (sample_num,), minval=ALPHA_RANGE[0], maxval=ALPHA_RANGE[1])
    yaw = jax.random.uniform(key3, (sample_num,), minval=YAW_RANGE[0], maxval=YAW_RANGE[1])
    pitch = jax.random.uniform(key4, (sample_num,), minval=PITCH_RANGE[0], maxval=PITCH_RANGE[1])
    p_true = jnp.stack([V_H, alpha, yaw, pitch], axis=1)
    # 正向模型输出真值
    y_true = batch_forward(p_true)
    # 叠加高斯噪声
    noise = jax.random.normal(key, y_true.shape) * NOISE_STD * jnp.mean(y_true, axis=0)
    y_obs = y_true + noise
    # 保存数据
    np.save("./output/data/p_true.npy", np.array(p_true))
    np.save("./output/data/y_obs.npy", np.array(y_true))
    np.save("./output/data/y_true.npy", np.array(y_true))
    return p_true, y_obs, y_true