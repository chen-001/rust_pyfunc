"""
Copula函数模块类型声明
=======================

提供4种常用Copula函数:
- Gaussian Copula: 对称依赖，无尾部依赖
- t-Copula: 对称依赖，有尾部依赖
- Clayton Copula: 下尾依赖（左尾），适合熊市因子共跌分析
- Gumbel Copula: 上尾依赖（右尾），适合牛市因子共涨分析
"""

from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray

# ==================== Gaussian Copula ====================

def gaussian_copula_cdf_py(u1: float, u2: float, rho: float) -> float:
    """
    Gaussian Copula CDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        rho: 相关系数 [-1, 1]

    返回:
        Copula CDF值
    """
    ...

def gaussian_copula_pdf_py(u1: float, u2: float, rho: float) -> float:
    """
    Gaussian Copula PDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        rho: 相关系数 [-1, 1]

    返回:
        Copula PDF值
    """
    ...

def gaussian_copula_cdf_batch(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    rho: float
) -> NDArray[np.floating]:
    """
    Gaussian Copula 批量CDF计算

    参数:
        u1: 均匀分布值数组 [0, 1]
        u2: 均匀分布值数组 [0, 1]
        rho: 相关系数 [-1, 1]

    返回:
        Copula CDF值数组
    """
    ...

def gaussian_copula_sample_py(rho: float, n: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gaussian Copula 采样

    参数:
        rho: 相关系数 [-1, 1]
        n: 样本数量

    返回:
        (u1, u2) 两个均匀分布样本数组
    """
    ...

def gaussian_copula_estimate(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Gaussian Copula 参数估计 (基于Kendall's tau)

    参数:
        x: 第一个变量数据
        y: 第二个变量数据

    返回:
        估计的相关系数rho
    """
    ...

# ==================== t-Copula ====================

def t_copula_cdf_py(u1: float, u2: float, rho: float, df: float) -> float:
    """
    t-Copula CDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        rho: 相关系数 [-1, 1]
        df: 自由度 (>0)

    返回:
        Copula CDF值
    """
    ...

def t_copula_pdf_py(u1: float, u2: float, rho: float, df: float) -> float:
    """
    t-Copula PDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        rho: 相关系数 [-1, 1]
        df: 自由度 (>0)

    返回:
        Copula PDF值
    """
    ...

def t_copula_cdf_batch(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    rho: float,
    df: float
) -> NDArray[np.floating]:
    """
    t-Copula 批量CDF计算

    参数:
        u1: 均匀分布值数组 [0, 1]
        u2: 均匀分布值数组 [0, 1]
        rho: 相关系数 [-1, 1]
        df: 自由度 (>0)

    返回:
        Copula CDF值数组
    """
    ...

def t_copula_sample_py(rho: float, df: float, n: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    t-Copula 采样

    参数:
        rho: 相关系数 [-1, 1]
        df: 自由度 (>0)
        n: 样本数量

    返回:
        (u1, u2) 两个均匀分布样本数组
    """
    ...

def t_copula_estimate(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    t-Copula 参数估计 (基于Kendall's tau)

    参数:
        x: 第一个变量数据
        y: 第二个变量数据

    返回:
        估计的相关系数rho
    """
    ...

# ==================== Clayton Copula ====================

def clayton_copula_cdf_py(u1: float, u2: float, theta: float) -> float:
    """
    Clayton Copula CDF计算 (下尾依赖)

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        theta: 依赖参数 (>0)

    返回:
        Copula CDF值
    """
    ...

def clayton_copula_pdf_py(u1: float, u2: float, theta: float) -> float:
    """
    Clayton Copula PDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        theta: 依赖参数 (>0)

    返回:
        Copula PDF值
    """
    ...

def clayton_copula_cdf_batch(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    theta: float
) -> NDArray[np.floating]:
    """
    Clayton Copula 批量CDF计算

    参数:
        u1: 均匀分布值数组 [0, 1]
        u2: 均匀分布值数组 [0, 1]
        theta: 依赖参数 (>0)

    返回:
        Copula CDF值数组
    """
    ...

def clayton_copula_sample_py(theta: float, n: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Clayton Copula 采样

    参数:
        theta: 依赖参数 (>0)
        n: 样本数量

    返回:
        (u1, u2) 两个均匀分布样本数组
    """
    ...

def clayton_copula_estimate(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Clayton Copula 参数估计 (基于Kendall's tau)

    参数:
        x: 第一个变量数据
        y: 第二个变量数据

    返回:
        估计的theta参数
    """
    ...

def clayton_lower_tail_dependence_py(theta: float) -> float:
    """
    Clayton Copula 下尾依赖系数

    参数:
        theta: 依赖参数 (>0)

    返回:
        下尾依赖系数 lambda_L = 2^(-1/theta)
    """
    ...

# ==================== Gumbel Copula ====================

def gumbel_copula_cdf_py(u1: float, u2: float, theta: float) -> float:
    """
    Gumbel Copula CDF计算 (上尾依赖)

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        theta: 依赖参数 (>=1)

    返回:
        Copula CDF值
    """
    ...

def gumbel_copula_pdf_py(u1: float, u2: float, theta: float) -> float:
    """
    Gumbel Copula PDF计算

    参数:
        u1: 均匀分布值 [0, 1]
        u2: 均匀分布值 [0, 1]
        theta: 依赖参数 (>=1)

    返回:
        Copula PDF值
    """
    ...

def gumbel_copula_cdf_batch(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    theta: float
) -> NDArray[np.floating]:
    """
    Gumbel Copula 批量CDF计算

    参数:
        u1: 均匀分布值数组 [0, 1]
        u2: 均匀分布值数组 [0, 1]
        theta: 依赖参数 (>=1)

    返回:
        Copula CDF值数组
    """
    ...

def gumbel_copula_sample_py(theta: float, n: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gumbel Copula 采样

    参数:
        theta: 依赖参数 (>=1)
        n: 样本数量

    返回:
        (u1, u2) 两个均匀分布样本数组
    """
    ...

def gumbel_copula_estimate(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Gumbel Copula 参数估计 (基于Kendall's tau)

    参数:
        x: 第一个变量数据
        y: 第二个变量数据

    返回:
        估计的theta参数
    """
    ...

def gumbel_upper_tail_dependence_py(theta: float) -> float:
    """
    Gumbel Copula 上尾依赖系数

    参数:
        theta: 依赖参数 (>=1)

    返回:
        上尾依赖系数 lambda_U = 2 - 2^(1/theta)
    """
    ...

# ==================== 工具函数 ====================

def copula_kendall_tau(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    计算Kendall's tau秩相关系数

    参数:
        x: 第一个变量数据
        y: 第二个变量数据

    返回:
        Kendall's tau值 [-1, 1]
    """
    ...

def to_uniform(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    将数据转换为均匀分布（经验CDF）

    参数:
        data: 原始数据

    返回:
        转换后的均匀分布值 [0, 1]
    """
    ...

def empirical_tail_dependence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    q: float = 0.1
) -> Dict[str, float]:
    """
    非参数估计尾部依赖（直接从数据计算，不假设分布）

    这是真正有意义的尾部依赖估计！

    参数:
        x: 第一个变量数据
        y: 第二个变量数据
        q: 分位数阈值，默认0.1表示最差10%

    返回:
        包含尾部依赖的字典:
        - lower_tail: 下尾依赖，P(Y也很差 | X很差)
        - upper_tail: 上尾依赖，P(Y也很好 | X很好)
        - quantile: 使用的分位数阈值
    """
    ...

def estimate_all_copulas(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    q: float = 0.1
) -> Dict[str, float]:
    """
    批量估计所有Copula参数（包含参数化和非参数方法）

    参数:
        x: 第一个变量数据
        y: 第二个变量数据
        q: 非参数尾部依赖的分位数阈值，默认0.1

    返回:
        包含所有参数估计的字典:

        【基础相关性指标】
        - kendall_tau: Kendall's tau秩相关系数

        【参数化Copula估计】
        - gaussian_rho: Gaussian Copula相关系数
        - t_rho: t-Copula相关系数
        - clayton_theta: Clayton Copula参数
        - gumbel_theta: Gumbel Copula参数
        - clayton_lower_tail_param: Clayton尾部依赖（假设数据服从Clayton）
        - gumbel_upper_tail_param: Gumbel尾部依赖（假设数据服从Gumbel）

        【非参数尾部依赖 - 真正有意义的！】
        - empirical_lower_tail: 下尾依赖，股灾时一起跌的概率
        - empirical_upper_tail: 上尾依赖，牛市时一起涨的概率
        - quantile: 使用的分位数阈值
    """
    ...

def copula_analysis(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    q: float = 0.1,
    df: float = None
) -> Dict[str, Dict[str, float]]:
    """
    Copula综合分析函数（五种模型拟合 + 拟合优度比较 + 尾部依赖分析）

    支持5种Copula模型:
    - Gaussian / t-Copula: 对称依赖，支持正负相关
    - Frank Copula: 对称依赖，天然支持正负相关，无尾部依赖
    - Clayton / Rotated-Clayton-90: 下尾依赖，负相关时自动使用90°旋转版本
    - Gumbel / Rotated-Gumbel-90: 上尾依赖，负相关时自动使用90°旋转版本

    当 kendall_tau <= 0 时，Clayton/Gumbel 自动切换为90°旋转版本以适配负相关。

    参数:
        x: 第一个变量数据（如ask_vol1）
        y: 第二个变量数据（如bid_vol1）
        q: 尾部依赖计算的分位数阈值，默认0.1（最极端10%）
        df: t-Copula的自由度，默认None表示自动优化

    返回:
        包含三个字典的结果:

        【params】参数信息
        - quantile:         用户指定的分位数阈值，用于经验尾部依赖计算
        - t_df_auto:        是否自动优化t-Copula自由度 (True/False)
        - t_df:             t-Copula自由度，取值范围 (0, +∞)
                            越小尾部越厚（极端共现事件概率越大），越大越接近正态
        - n_samples:        有效样本量

        【estimates】从数据估计的结果

        Gaussian Copula:
        - gaussian_rho:             由 tau 通过 sin(tau * π/2) 转换得到，取值 [-1, 1]
                                    绝对值越大相关越强，0 = 独立
        - gaussian_log_likelihood:  对数似然，取值 (-∞, +∞)，越大拟合越好

        t-Copula:
        - t_rho:                    与 gaussian_rho 相同（同源于 tau 转换），取值 [-1, 1]
        - t_log_likelihood:         对数似然，取值 (-∞, +∞)，越大拟合越好
                                    若显著大于 gaussian_log_likelihood，说明厚尾效应明显

        Frank Copula:
        - frank_theta:              Frank参数，取值 (-∞, +∞)
                                    > 0 正相关，< 0 负相关，绝对值越大相关越强，→0 趋近独立
                                    Frank 无尾部依赖，适合描述"中间段有相关但极端不联动"
        - frank_log_likelihood:     对数似然，取值 (-∞, +∞)，越大拟合越好

        Clayton Copula (正相关时) / Rotated-Clayton-90 (负相关时):
        - clayton_type:             "Clayton"(tau>0) 或 "Rotated-Clayton-90"(tau<=0)
        - clayton_theta:            Clayton参数，取值 (0, +∞)
                                    越大依赖越强，→0 趋近独立
                                    注意：负相关时参数由 |tau| 估计，仍为正值
        - clayton_log_likelihood:   对数似然，取值 (-∞, +∞)，越大拟合越好
        - clayton_tail_dependence:  理论尾部依赖系数 = 2^(-1/θ)，取值 [0, 1]
                                    原版Clayton: 衡量同向下尾（X和Y同时极小）的联动概率
                                    Rotated-90: 衡量交叉尾部（X极大时Y极小）的联动概率
                                    越接近1联动越强，越接近0越独立

        Gumbel Copula (正相关时) / Rotated-Gumbel-90 (负相关时):
        - gumbel_type:              "Gumbel"(tau>0) 或 "Rotated-Gumbel-90"(tau<=0)
        - gumbel_theta:             Gumbel参数，取值 [1, +∞)
                                    越大依赖越强，=1 时为独立
                                    注意：负相关时参数由 |tau| 估计，仍 >= 1
        - gumbel_log_likelihood:    对数似然，取值 (-∞, +∞)，越大拟合越好
        - gumbel_tail_dependence:   理论尾部依赖系数 = 2 - 2^(1/θ)，取值 [0, 1]
                                    原版Gumbel: 衡量同向上尾（X和Y同时极大）的联动概率
                                    Rotated-90: 衡量交叉尾部（X极小时Y极大）的联动概率
                                    越接近1联动越强，越接近0越独立

        非参数经验尾部依赖:
        - empirical_lower_tail:       P(Y<q分位 | X<q分位)，取值 [0, 1]
                                      衡量两者同时处于极低分位的概率，越大说明同跌联动越强
        - empirical_upper_tail:       P(Y>1-q分位 | X>1-q分位)，取值 [0, 1]
                                      衡量两者同时处于极高分位的概率，越大说明同涨联动越强
        - empirical_cross_lower_tail: P(Y<q分位 | X>1-q分位)，取值 [0, 1]
                                      X极大时Y极小的概率，负相关核心指标，越大反向联动越强
        - empirical_cross_upper_tail: P(X<q分位 | Y>1-q分位)，取值 [0, 1]
                                      X极小时Y极大的概率，负相关核心指标，越大反向联动越强

        【evaluation】模型评价和建议
        - gaussian_aic / t_aic / frank_aic / clayton_aic / gumbel_aic:
                                    AIC信息准则 = 2k - 2*loglik，取值 (-∞, +∞)
                                    越小（越负）拟合越好，不同模型间可直接比较
        - best_copula:              AIC最小的模型名称
        - best_aic:                 最佳AIC值
        - best_avg_loglik:          最佳模型的每样本平均对数似然，取值 [0, +∞)
                                    = best_loglik / n_samples
                                    独立时 = 0，越大说明依赖越强
                                    跨例子可直接比较：无论最佳模型是哪个族、样本量多少
                                    例如 A=0.015 vs B=0.032 → B的依赖关系更强
        - tail_asymmetry:           尾部不对称度，取值 (-1, 1)
                                    正相关时 = empirical_lower - empirical_upper
                                        > 0 共跌型（熊市联动更强），< 0 共涨型（牛市联动更强）
                                    负相关时 = cross_lower - cross_upper
                                        > 0 卖强买弱联动更显著，< 0 买强卖弱联动更显著
        - tail_type:                尾部类型的文字描述
                                    正相关: "共跌型（熊市联动更强）" / "共涨型（牛市联动更强）"
                                           / "对称尾部依赖" / "近独立（极端行情不联动）"
                                    负相关: "负相关-反向下尾型（X涨时Y跌联动更强）"
                                           / "负相关-反向上尾型（X跌时Y涨联动更强）"
                                           / "负相关-对称反向联动"
                                           / "负相关-弱联动（极端行情反向性不显著）"
    """
    ...
