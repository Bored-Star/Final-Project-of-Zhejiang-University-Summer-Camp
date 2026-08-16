# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def plot_error_bar(p_true, p_est, save_path):
    """四参数平均相对误差柱状图"""
    rel_err = np.abs((p_est - p_true) / (p_true + 1e-6)).mean(axis=0)
    labels = ["风速Vh", "风切α", "偏航", "桨距"]
    plt.figure(figsize=(8,4))
    plt.bar(labels, rel_err)
    plt.ylabel("平均相对误差")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_curve(iter_list, loss_list, save_path):
    """损失收敛曲线"""
    plt.figure(figsize=(8,4))
    plt.plot(iter_list, loss_list)
    plt.xlabel("迭代步数")
    plt.ylabel("损失值")
    plt.savefig(save_path, dpi=300)
    plt.close()