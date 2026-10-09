import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

'''
scipy.optimize.curve_fit(f, degreeXdata, degreeYdata, p0=None, sigma=None, absolute_sigma=False, check_finite=True,
                        bounds=(- inf, inf), method=None, jac=None, **kwargs)

Assumes degreeYdata = f(degreeXdata, *params) + eps.

Notes:
With method='lm', the algorithm uses the Levenberg-Marquardt algorithm through leastsq. 
Note that this algorithm can only deal with unconstrained problems.
'''


def CurveFit(xdata, ydata, distribution):
    popt = None
    pcov = None
    ydist = None

    if distribution == 'PowerLaw':
        try:
            popt, pcov = curve_fit(PowerLaw, xdata, ydata, maxfev=50000)
            ydist = [PowerLaw(i, popt[0], popt[1]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'Exponential':
        try:
            popt, pcov = curve_fit(Exponential, xdata, ydata)
            ydist = [Exponential(i, popt[0], popt[1]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'Lognormal':
        try:
            popt, pcov = curve_fit(Lognormal, xdata, ydata, bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)))
            ydist = [Lognormal(i, popt[0], popt[1], popt[2]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'TrunPowerLaw':
        try:
            popt, pcov = curve_fit(TrunPowerLaw, xdata, ydata)
            ydist = [TrunPowerLaw(i, popt[0], popt[1], popt[2]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'Gamma':
        try:
            popt, pcov = curve_fit(Gamma, xdata, ydata, maxfev=50000)
            ydist = [Gamma(i, popt[0], popt[1], popt[2]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'InvGaussian':
        try:
            popt, pcov = curve_fit(InvGaussian, xdata, ydata, bounds=((-np.inf, 0, 0), (np.inf, np.inf, np.inf)),
                                   maxfev=50000)
            ydist = [InvGaussian(i, popt[0], popt[1], popt[2]) for i in xdata]
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')
    else:
        print('No aviliable distribution:'+distribution)

    return popt, pcov, ydist


def Exponential(x, a, b):
    return a * np.exp(-b * x)


def PowerLaw(x, a, b):
    return a * x ** (-b)


def TrunPowerLaw(x, a, b, c):
    return a * x ** (-b) * np.e ** (-c * x)


def Lognormal(x, c, mu, sigma):
    b1 = c / (np.sqrt(2 * np.pi) * sigma * x)
    b2 = np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    return b1 * b2


def Gamma(x, c, k, theta):
    b1 = c * x ** (k - 1) * np.exp(-x / theta)
    b2 = theta ** k * gamma(k)
    return b1 / (b2 + 1e-10)


def InvGaussian(x, c, lamda, mu):
    b1 = c * (lamda / (2 * np.pi * x ** 3)) ** 0.5
    b2 = np.exp(-(lamda * (x - mu) ** 2) / (2 * mu ** 2 * x))
    return b1 * b2


def loglikelihood(xdata, ydata, distribution):
    ydist = None
    yLL = None

    if distribution == 'PowerLaw':
        try:
            popt, pcov = curve_fit(PowerLaw, xdata, ydata, maxfev=50000)
            ydist = [np.log(PowerLaw(i, popt[0], popt[1])) for i in xdata]
            yLL = np.sum(ydist)
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'Exponential':
        try:
            popt_e, pcov_e = curve_fit(Exponential, xdata, ydata)
            ydist = [np.log(Exponential(i, popt_e[0], popt_e[1])) for i in xdata]
            yLL = np.sum(ydist)
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'Lognormal':
        try:
            popt_l, pcov_l = curve_fit(Lognormal, xdata, ydata, maxfev=50000,
                                       bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)))
            ydist = [np.log(Lognormal(i, popt_l[0], popt_l[1], popt_l[2])) for i in xdata]
            yLL = np.sum(ydist)

        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')

    elif distribution == 'TrunPowerLaw':
        try:
            popt_tp, pcov_tp = curve_fit(TrunPowerLaw, xdata, ydata,bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, 0.1)), maxfev=50000)
            ydist = [np.log(TrunPowerLaw(i, popt_tp[0], popt_tp[1], popt_tp[2])) for i in xdata]
            yLL = np.sum(ydist)
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')


    elif distribution == 'Gamma':
        try:
            popt_g, pcov_g = curve_fit(Gamma, xdata, ydata, maxfev=50000)
            ydist = [np.log(Gamma(i, popt_g[0], popt_g[1], popt_g[2])) for i in xdata]
            yLL = np.sum(ydist)
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')



    elif distribution == 'InvGaussian':
        try:
            popt_i, pcov_i = curve_fit(InvGaussian, xdata, ydata, bounds=((-np.inf, 0, 0), (np.inf, np.inf, np.inf)),
                                       maxfev=50000)
            ydist = [np.log(InvGaussian(i, popt_i[0], popt_i[1], popt_i[2])) for i in xdata]
            yLL = np.sum(ydist)
        except RuntimeError:
            print(distribution + ':No optimal fit found', end=' ')
            ydist = None
            yLL = None
    else:
        print('No aviliable distribution' + distribution)

    return ydist, yLL


def LLRtest(xdata, ydata, distribution1, distribution2):
    ydist1, LL1 = loglikelihood(xdata, ydata, distribution1)
    ydist2, LL2 = loglikelihood(xdata, ydata, distribution2)
    if ydist1 == None or ydist2 == None:
        return 0, 0
    LLR = LL1 - LL2
    # calcu p-value
    n = len(xdata)
    ll1_a = LL1 / n
    ll2_a = LL2 / n
    b1 = 0
    for i in range(n):
        b1 = ((ydist1[i] - ll1_a) - (ydist2[i] - ll2_a)) ** 2 + b1
    sigma2 = b1 / n

    p = math.erfc(LLR / np.sqrt(2 * n * sigma2))

    # LLR大于0，则表明
    return LLR, p


def R2_Mean_SD_Calau(xdata, ydata, distribution):

    popt, pcov, yhat= CurveFit(xdata, ydata,distribution)

    if  popt is None:
        print('No optimal fit to '+distribution)
        R2, RMSE, SD=-1, -1, -1
    else:
        n=len(ydata)
        ybar=np.sum(ydata)/n
        'R2'
        Res=[(ydata[i]-yhat[i])**2 for i in range(n)]

        SSRes=np.sum(Res)
        Tot=[(ydata[i]-ybar)**2 for i in range(n)]
        SSTot=np.sum(Tot)

        R2=1-SSRes/SSTot
        'RMSE'
        RMSE=np.sqrt(SSRes/n)
        'SD'
        SD = np.sqrt(np.diag(pcov))

    return R2, RMSE, SD



def LogBinning(degrees,degreefreq):
    degreefreq = [freq / sum(degreefreq) for freq in degreefreq]  # 试一下看看
    #plt.loglog(degrees, degreefreq, 'r.', alpha=0.3, label='linear binning')
    binsEnd = np.log2(max(degrees))
    binsEnd = int(binsEnd) + 1
    lenDegFreq = len(degreefreq)
    # print(lenDegFreq)
    zeroList = [0 for x in range(0, int(2 ** binsEnd - max(degrees)))]
    degreefreq = degreefreq + zeroList

    logBinningSpace = []
    for i in range(0, binsEnd):
        p = 2 ** i
        binStart = p
        binend = binStart + p - 1
        logBinningSpace.append([binStart, binend])

    # print(logBinningSpace)
    xdatabinning = []
    ydatabinning = []

    for binList in logBinningSpace:
        Start = binList[0]
        End = binList[1]
        xdataAvg = (End + Start) / 2
        xdatabinning.append(xdataAvg)
        binningList = degreefreq[Start - 1:End]
        nodeNumber = End - Start + 1
        # print(nodeNumber)
        binListSum = sum(binningList)
        ydatabinning.append(binListSum / nodeNumber)

    return xdatabinning, ydatabinning






if __name__ == "__main__":
    # data from LogbinningV2.py
    # xdata1 = [1.0, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5,
    #           2, 3, 4, 5, 6, 7, 8, 9, 10]
    # degreeYdata = [0.4808376135914658, 0.050967996839193994, 0.009976293954958514, 0.002173054128802845,
    #          0.0002469379691821414, 6.173449229553535e-05, 6.173449229553536e-06,
    #          0.07269853812722245, 0.029237455551165546, 0.01698933227973133, 0.009482418016594232, 0.007111813512445673,
    #          0.006321612011062821, 0.004741209008297116, 0.0039510075069142635, 0.0019755037534571317]
    '''
    xdata1=[2.5, 5.5, 11.5, 23.5, 47.5, 95.5]
    ydata = [0.3349834983498348, 0.06641914191419139, 0.006084983498349832, 0.0007219471947194716,
                         0.0001031353135313531, 1.2891914191419137e-05]
    norm = sum(ydata)
    ydata1 = [i / norm for i in ydata]
    # plt.scatter(xdata1, ydata1)

    #DATA2
    # xdata1=[1.0, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5, 191.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    # degreeYdata=[0.5702040816326527, 0.06826530612244892, 0.028571428571428553, 0.010382653061224482, 0.003915816326530609, 0.0007079081632653055, 0.00013711734693877537, 1.4349489795918356e-05, 0.08326530612244891, 0.05326530612244894, 0.040816326530612214, 0.03163265306122447, 0.022244897959183656, 0.019591836734693863, 0.015918367346938765, 0.013061224489795908, 0.011836734693877542, 0.008775510204081627, 0.008571428571428565, 0.010408163265306114, 0.006734693877551015, 0.0077551020408163215, 0.005510204081632649, 0.00571428571428571, 0.0044897959183673435, 0.005306122448979588, 0.005306122448979588, 0.004897959183673466, 0.005918367346938771, 0.0042857142857142825, 0.0042857142857142825, 0.0038775510204081608, 0.0022448979591836718, 0.002653061224489794, 0.002448979591836733, 0.0018367346938775496, 0.0022448979591836718, 0.0016326530612244886, 0.0014285714285714275, 0.0018367346938775496, 0.0006122448979591832, 0.0016326530612244886, 0.0012244897959183664, 0.0018367346938775496, 0.0006122448979591832, 0.0012244897959183664, 0.0006122448979591832]
    # norm = sum(degreeYdata)
    # ydata1 = [i / norm for i in degreeYdata]

    # xdata1=range(1,100)
    # ydata1=[PowerLaw(x, 1, 3) for x in xdata1]
    plt.scatter(xdata1, ydata1)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print(LLRtest(xdata1, ydata1, 'PowerLaw', 'Exponential'))
    print(LLRtest(xdata1, ydata1, 'PowerLaw', 'Lognormal'))
    print(LLRtest(xdata1, ydata1, 'PowerLaw', 'InvGaussian'))
    print(LLRtest(xdata1, ydata1, 'TrunPowerLaw', 'PowerLaw'))
    print(LLRtest(xdata1, ydata1, 'PowerLaw', 'TrunPowerLaw'))
    print(LLRtest(xdata1, ydata1, 'PowerLaw', 'Gamma'))

    R2, RMSE, SD=R2_Mean_SD_Calau(xdata1, ydata1, 'PowerLaw')
    print('R2, RMSE, SD=',R2, RMSE, SD)
    '''

