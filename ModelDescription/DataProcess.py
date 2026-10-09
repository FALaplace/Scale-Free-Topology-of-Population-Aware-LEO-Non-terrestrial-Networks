import matplotlib.pyplot as plt
import numpy as np
from ModelDiscriptionFunc import Exponential, Gamma, InvGaussian, LLRtest, Lognormal, PowerLaw, TrunPowerLaw
from scipy.optimize import curve_fit, minimize
from pathlib import Path
import pickle

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})


def readFiles(file_path):
    line1 = []
    with open(file_path) as f:
        for line in f.readlines():
            line1.append(line.split(',')[0:-1])

    for i in range(len(line1)):
        for j in range(len(line1[i])):
            line1[i][j] = float(line1[i][j])
    degree, degreeFreq = line1[0], line1[1]
    return degree[1:], degreeFreq[1:]


# discretePowerLaw
def discreteFormula(xdata):
    xmin = min(xdata)
    n = len(xdata)
    sum = 0
    for i in range(n):
        sum = sum + np.log(xdata[i] / (xmin - 0.5))
    sum = n / sum
    alpha = 1 + sum
    return alpha


def discretePowerLaw(x, alpha, xdata):
    xmin = min(xdata)
    zeta = 0
    for i in range(100):
        zeta = zeta + (i + xmin) ** (-alpha)
    return (x ** (-alpha)) / zeta


def lik_discretePowerLaw(parameters, xdata):
    # zeta
    xmin = min(xdata)
    n = len(xdata)
    zeta = 0
    for i in range(1000):
        zeta = zeta + (i + xmin) ** (-parameters)
    b = 0
    for i in range(n):
        b = b + np.log(xdata[i])
    L = -n * np.log(zeta) - parameters * b
    return -L


def maxlikhood_DiscretePowerlaw(xdata, xdata1):
    bnds = (0, None)  # 2.2167442396767294
    lik_model_DiscretePowerlaw = minimize(lik_discretePowerLaw, np.array(2.2), args=(xdata,), method='Nelder-Mead')
    yDiscPowerlaw1 = [discretePowerLaw(i, lik_model_DiscretePowerlaw.x[0], xdata) for i in xdata1]
    norm = sum(yDiscPowerlaw1)
    yDiscPowerlaw = [i / norm for i in yDiscPowerlaw1]
    return lik_model_DiscretePowerlaw.x, yDiscPowerlaw


# Ecponential
def maxlikhood_Exponential(x0_e, bnds, xdata, xdata1):
    # bnds=((3.9, 4), (1.4, 1.5)) #popt_e= [3.92766691 1.49200849]
    lik_model_Exponential = minimize(lik_Exponential, x0_e, args=(xdata,), bounds=bnds, method='SLSQP')
    yExponent1 = [Exponential(i, lik_model_Exponential.x[0], lik_model_Exponential.x[1]) for i in xdata1]
    norm = sum(yExponent1)
    yExponent = [i / norm for i in yExponent1]
    return lik_model_Exponential.x, yExponent


def lik_Exponential(parameters, xdata):
    a = parameters[0]
    b = parameters[1]

    y_exp = 0
    for i in np.arange(0, len(xdata)):
        y_exp = y_exp + np.log(Exponential(xdata[i], a, b))
    return -y_exp


# Lognormal
def maxlikhood_Lognormal(x0_l, bnds, xdata, xdata1):
    # 4.418064819334306 0.6271978931447243 0.8058173088128446
    lik_model_Lognormal = minimize(lik_Lognormal, x0_l, args=(xdata,), bounds=bnds, method='L-BFGS-B')

    yLognormal1 = [Lognormal(i, lik_model_Lognormal.x[0], lik_model_Lognormal.x[1], lik_model_Lognormal.x[2]) for i in
                   xdata1]
    norm = sum(yLognormal1)
    yLognormal = [i / norm for i in yLognormal1]
    return lik_model_Lognormal.x, yLognormal


def lik_Lognormal(parameters, xdata):
    c = parameters[0]
    mu = parameters[1]
    sigma = parameters[2]

    y_exp = 0
    for i in np.arange(0, len(xdata)):
        y_exp = y_exp + np.log(Lognormal(xdata[i], c, mu, sigma))
    y_exp = -y_exp
    return y_exp


# TrunPowerLaw
def maxlikhood_TrunPowerLaw(x0_tp, bnds, xdata, xdata1):
    lik_model_TrunPowerLaw = minimize(lik_TrunPowerLaw, np.array([0.35, -1.01, 0.03]), args=(xdata,), bounds=bnds,
                                      method='L-BFGS-B')  # L-BFGS-B / BFGS

    yTrunPowerLaw1 = [
        TrunPowerLaw(i, lik_model_TrunPowerLaw.x[0], lik_model_TrunPowerLaw.x[1], lik_model_TrunPowerLaw.x[2]) for i in
        xdata1]
    norm = sum(yTrunPowerLaw1)
    yTrunPowerLaw = [i / norm for i in yTrunPowerLaw1]
    return lik_model_TrunPowerLaw.x, yTrunPowerLaw


def lik_TrunPowerLaw(parameters, xdata):
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    y_exp = 0
    for i in range(len(xdata)):
        y_exp = y_exp + np.log(TrunPowerLaw(xdata[i], a, b, c))
    y_exp = -y_exp
    return y_exp


# Gamma
def maxlikhood_Gamma(x0_g, bnds, xdata, xdata1):
    lik_model_Gamma = minimize(lik_Gamma, x0_g, args=(xdata,), bounds=bnds, method='L-BFGS-B')
    yGamma1 = [Gamma(i, lik_model_Gamma.x[0], lik_model_Gamma.x[1], lik_model_Gamma.x[2]) for i in xdata1]
    norm = sum(yGamma1)
    yGamma = [i / norm for i in yGamma1]
    return lik_model_Gamma.x, yGamma


def lik_Gamma(parameters, xdata):
    c = parameters[0]
    k = parameters[1]
    theta = parameters[2]
    y_exp = 0
    for i in range(len(xdata)):
        y_exp = y_exp + np.log(Gamma(xdata[i], c, k, theta))
    y_exp = -y_exp
    return y_exp


# InvGaussian
def lik_InvGaussian(parameters, xdata):
    c = parameters[0]
    lamda = parameters[1]
    mu = parameters[2]
    y_exp = 0
    for i in range(len(xdata)):
        y_exp = y_exp + np.log(InvGaussian(xdata[i], c, lamda, mu))
    y_exp = -y_exp
    return y_exp


def maxlikhood_InvGaussian(x0_i, bnds, xdata, xdata1):
    lik_model_InvGaussian = minimize(lik_InvGaussian, x0_i, args=(xdata,), bounds=bnds,
                                     method='SLSQP')  # Nelder-Mead/ L-BFGS-B / SLSQP
    yInvGauss1 = [InvGaussian(i, lik_model_InvGaussian.x[0], lik_model_InvGaussian.x[1], lik_model_InvGaussian.x[2]) for
                  i in xdata1]
    norm = sum(yInvGauss1)
    yInvGauss = [i / norm for i in yInvGauss1]
    return lik_model_InvGaussian.x, yInvGauss


def analyze_distribution(xNum, SynCityNum, linear_point_count=50):
    """分析一组配置，返回绘图数据、拟合参数和原流程的模型比较结果。

    xNum 为卫星网格边长，卫星数为 xNum**2；SynCityNum 为终端数。
    linear_point_count 沿用原脚本的低度区截取方式，默认取前 50 个点。
    本函数保留 Figure 6 的原有描述性拟合方法，不创建图形。
    """
    if xNum <= 0 or int(xNum) != xNum:
        raise ValueError('xNum must be a positive integer')
    if SynCityNum <= 0 or int(SynCityNum) != SynCityNum:
        raise ValueError('SynCityNum must be a positive integer')
    if linear_point_count < 0 or int(linear_point_count) != linear_point_count:
        raise ValueError('linear_point_count must be a nonnegative integer')
    xNum, SynCityNum = int(xNum), int(SynCityNum)
    linear_point_count = int(linear_point_count)
    GSNUMBER = xNum * xNum
    data_dir = Path(__file__).resolve().parent.parent / 'heatmap (LLR)'
    filename = f'GS{GSNUMBER}_CITY{SynCityNum}.txt'
    degrees, degreeFreqStart4 = readFiles(data_dir / 'scaleFreeProveData(whole)' / filename)
    degreeBinning, degreeFreqBinning = readFiles(data_dir / 'scaleFreeProveData' / filename)

    # 沿用原脚本：合并对数分箱点与低度区原始点，再归一化。
    degreeXdata = degreeBinning + degrees[:linear_point_count]
    degreeYdata1 = degreeFreqBinning + degreeFreqStart4[:linear_point_count]
    if (len(degreeXdata) != len(degreeYdata1) or not degreeXdata
            or not np.all(np.isfinite(degreeXdata)) or min(degreeXdata) <= 0
            or not np.all(np.isfinite(degreeYdata1)) or min(degreeYdata1) < 0
            or sum(degreeYdata1) <= 0):
        raise ValueError(f'Invalid degree-frequency data in {filename}')
    norm = sum(degreeYdata1)
    degreeYdata = [i / norm for i in degreeYdata1]
    xdata1 = degreeXdata

    # 保留原先按频率构造拟合样本的规则。
    xdata = []
    Num = 10000
    for i in range(len(degreeXdata)):
        for j in range(int(Num * degreeYdata[i])):
            xdata.append(degreeXdata[i])
    if not xdata:
        raise ValueError('The reconstructed fitting sample is empty')

    # 最小二乘初值：拟合方法、边界和迭代设置与原脚本一致。
    popt, _ = curve_fit(PowerLaw, degreeXdata, degreeYdata, maxfev=50000)
    popt_tp, _ = curve_fit(
        TrunPowerLaw, degreeXdata, degreeYdata,
        bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)), maxfev=50000)
    popt_e, _ = curve_fit(Exponential, degreeXdata, degreeYdata)
    popt_l, _ = curve_fit(
        Lognormal, degreeXdata, degreeYdata, maxfev=50000,
        bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)))
    popt_g, _ = curve_fit(Gamma, degreeXdata, degreeYdata, maxfev=50000)
    popt_i, _ = curve_fit(
        InvGaussian, degreeXdata, degreeYdata,
        bounds=((-np.inf, 0, 0), (np.inf, np.inf, np.inf)), maxfev=5000)

    # 最大似然拟合：样本与绘图横坐标显式传入，配置之间不共享全局数据。
    mle_p, ydata_p = maxlikhood_DiscretePowerlaw(xdata, xdata1)
    bnds_e = ((popt_e[0] - 0.2, popt_e[0] + 0.2),
              (popt_e[1] - 0.2, popt_e[1] + 0.2))
    mle_e, ydata_e = maxlikhood_Exponential(popt_e, bnds_e, xdata, xdata1)
    bnds_l = ((popt_l[0] - 0.2, popt_l[0] + 0.2),
              (popt_l[1] - 0.2, popt_l[1] + 0.2),
              (popt_l[2] - 0.2, popt_l[2] + 0.2))
    mle_l, ydata_l = maxlikhood_Lognormal(popt_l, bnds_l, xdata, xdata1)
    bnds_tp = ((popt_tp[0] - 0.01, popt_tp[0] + 0.01),
               (popt_tp[1] - 0.05, popt_tp[1] + 0.05),
               (popt_tp[2] - 0.001, popt_tp[2] + 0.1))
    mle_tp, ydata_tp = maxlikhood_TrunPowerLaw(popt_tp, bnds_tp, xdata, xdata1)
    bnds_g = ((popt_g[0] - 5e3, popt_g[0] + 5e3),
              (popt_g[1] - 0.1, popt_g[1] + 0.1),
              (popt_g[2] - 0.5, popt_g[2] + 0.5))
    mle_g, ydata_g = maxlikhood_Gamma(popt_g, bnds_g, xdata, xdata1)
    bnds_i = ((popt_i[0] - 0.1, popt_i[0] + 0.1),
              (popt_i[1] - 1e-5, popt_i[1] + 1e-5),
              (popt_i[2] - 1e-2, popt_i[2] + 1e-2))
    mle_i, ydata_i = maxlikhood_InvGaussian(popt_i, bnds_i, xdata, xdata1)

    fits = {
        'PowerLaw': (popt, mle_p, ydata_p),
        'TrunPowerLaw': (popt_tp, mle_tp, ydata_tp),
        'Exponential': (popt_e, mle_e, ydata_e),
        'Lognormal': (popt_l, mle_l, ydata_l),
        'InvGaussian': (popt_i, mle_i, ydata_i),
        'Gamma': (popt_g, mle_g, ydata_g),
    }
    comparison_pairs = [
        ('PowerLaw', 'Exponential'), ('PowerLaw', 'Lognormal'),
        ('PowerLaw', 'InvGaussian'), ('TrunPowerLaw', 'PowerLaw'),
        ('PowerLaw', 'TrunPowerLaw'), ('PowerLaw', 'Gamma'),
        ('TrunPowerLaw', 'Lognormal'), ('TrunPowerLaw', 'Gamma'),
        ('TrunPowerLaw', 'InvGaussian'),
    ]
    comparisons = {
        f'{first} vs {second}': LLRtest(degreeXdata, degreeYdata, first, second)
        for first, second in comparison_pairs
    }
    return {
        'satellite_count': GSNUMBER,
        'terminal_count': SynCityNum,
        'linear_point_count': linear_point_count,
        'degree': np.asarray(degreeXdata),
        'frequency': np.asarray(degreeYdata),
        'binned_degree': np.asarray(degreeBinning),
        'binned_frequency': np.asarray(degreeFreqBinning),
        'least_squares_parameters': {name: fit[0] for name, fit in fits.items()},
        'mle_parameters': {name: fit[1] for name, fit in fits.items()},
        'fitted_frequency': {name: np.asarray(fit[2]) for name, fit in fits.items()},
        'discrete_alpha': discreteFormula(xdata),
        'comparisons': comparisons,
    }


def plot_distribution(result, ax=None):
    """只使用 analyze_distribution 的结果绘图；返回 Figure 和主坐标轴。

    ax 可传入六宫格中的一个子图；不传时创建单独图窗。
    图例、坐标、标记和 inset 的样式均集中在此函数中修改。
    """
    N = result["satellite_count"]
    N_T = result["terminal_count"]
    degree = result['degree']
    frequency = result['frequency']
    curves = result['fitted_frequency']

    ax.loglog(degree, frequency, 'r.', alpha=0.3, label='empirical data')
    ax.loglog(degree, curves['PowerLaw'], 'g.', alpha=0.6, label='powerlaw', zorder=10)
    ax.loglog(degree, curves['TrunPowerLaw'], 'b.', alpha=0.3, label='trun-powerlaw')
    ax.loglog(degree, curves['Exponential'], 'c.', alpha=0.3, label='Exponential')
    ax.loglog(degree, curves['Lognormal'], 'm.', alpha=0.3, label='Lognormal')
    ax.loglog(degree, curves['InvGaussian'], 'k.', alpha=0.3, label='InvGaussian')
    ax.scatter(degree, curves['Gamma'], color='orange', marker='.', alpha=0.5, label='Gamma')
    ax.set_ylim([1e-20, 5e2])
    if N == 900:
        ax.legend(loc='lower right', fontsize=15)
    if N in [16900, 14400]:
        ax.set_xlabel('Degree', fontsize=15)
    if N_T in [4000, 80000]:
        ax.set_ylabel('Frequency', fontsize=15)
    ax.set_title(fr"$N$ = {N}, $N_T$ = {N_T}", fontsize=16)

    # 使用子图内部坐标，保证单图和六宫格都可调用。
    inset = ax.inset_axes([.10, .12, .42, .32])
    inset.loglog(degree, frequency, 'r.', markersize=5, alpha=0.3)
    inset.loglog(degree, curves['PowerLaw'], 'g.', markersize=5, alpha=0.6, zorder=10)
    inset.loglog(degree, curves['InvGaussian'], 'k.', markersize=5, alpha=0.3)
    inset.loglog(degree, curves['TrunPowerLaw'], 'b.', markersize=5, alpha=0.3)
    inset.set_ylim([1e-6, 5e-1])
    return ax.figure, ax


if __name__ == '__main__':
    results_path = Path(__file__).resolve().parent / 'Results' / 'results.pkl'
    configurations = [(13, 4000), (20, 8000), (30, 10000), (130, 80000), (120, 100000), (130, 200000)]
    linear_point_count = 50

    # results = []
    # for xNum, SynCityNum in configurations:
    #     print(f'Analyzing N={xNum ** 2}, N_T={SynCityNum} ...', flush=True)
    #     result = analyze_distribution(xNum, SynCityNum, linear_point_count)
    #     results.append(result)
    #     for pair, statistics in result['comparisons'].items():
    #         print(pair, statistics)
    # with results_path.open('wb') as file:
    #     pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    with results_path.open('rb') as file:
        results = pickle.load(file)

    columns = 3
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    for ax, result in zip(axes.flat, results):
        plot_distribution(result, ax=ax)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()
