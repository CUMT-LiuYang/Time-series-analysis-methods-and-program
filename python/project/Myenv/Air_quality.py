import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis
from scipy.integrate import odeint
import statistics
from scipy.stats import lognorm
import math
import string
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import ttest_ind
from datetime import timedelta
import scipy.signal
import scipy.stats as st
# Using statmodels: Subtracting the Trend Component. https://www.machinelearningplus.com/time-series/time-series-analysis-python/
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.special import gamma, digamma
from scipy import optimize
from PyEMD import EMD
#increase the font size for plotting
plt.rc('font', size=16)

# 引用徐州空气质量监测中的“四种气体”作为时间序列分析数据（2023年1月至2023年12月）
# Import the "Four Gases" in Xuzhou Air Quality Monitoring as time series analysis data(From January 2023 to December 2023)
data = pd.read_csv("./data/2023_Xuzhou_Air_Indicators_Data_Every_One_Hour.csv")

# 将字符串格式的日期转换为 datetime 对象
# Convert a date in string format to a datetime object
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%dT%H:%M:%SZ")

# In[ ]:

# 定义颜色
# Define colors
current_palette = sns.color_palette()
COColor = current_palette[3]
NO2Color = current_palette[0]
O3Color = current_palette[2]
SO2Color = current_palette[4]



# 过滤获取各气体数据
# Filter to obtain the data of each gas
gas_data = {
    'CO': data.filter(like="CO (mg/m³)"),
    'NO2': data.filter(like="NO2 (µg/m³)"),
    'O3': data.filter(like="O3 (µg/m³)"),
    'SO2': data.filter(like="SO2 (µg/m³)")
}




def gas_import(gas_name):
    return gas_data[gas_name]

def gas_curves(data, gas_name, ylabel, exportName, axis, label_no):
    axis.plot(data['Date'], gas_data[gas_name], alpha=0.7, label=gas_name, color=eval(gas_name+"Color"))
    axis.set_xlabel('Date',fontsize=16)
    axis.tick_params(axis='x', labelsize=16)
    #plt.xticks(rotation=45)  # 将横坐标值旋转45度显示
    axis.set_ylabel(ylabel)
    axis.text(0.05, 0.9, string.ascii_lowercase[label_no], transform=axis.transAxes, size=16, weight='bold')
    axis.legend()


# In[ ]:

fig, axs = plt.subplots(2,2,figsize=(16,12))
ax1 = axs[0, 0]
ax2 = axs[1, 0]
ax3 = axs[1, 1]
ax4 = axs[0, 1]


gas_curves(data, 'CO', 'CO (mg/m³)', 'exportName', ax1, 0)
gas_curves(data, 'NO2', 'NO2 (µg/m³)', 'exportName', ax2, 1)
gas_curves(data, 'O3', 'O3 (µg/m³)', 'exportName', ax3, 2)
gas_curves(data, 'SO2', 'SO2 (µg/m³)', 'exportName', ax4, 3)
fig.autofmt_xdate(rotation=45)
#fig.suptitle("Air Quality Gas Concentrations", fontsize=18)
fig.align_ylabels(axs[:, 1])
plt.savefig('Sup_pics/Fig1.pdf', bbox_inches='tight')
plt.show()



# In[ ]:

# 数据截取（可按数据量截取，也可编写日期处理函数按日期截取）
# Data Interception (can be intercepted by data volume, or you can write a date processing function to intercept by date)
def filter_gas_data(gas_data, gas_name, start_idx, length):
    """
    Filter gas data based on gas name and index range.

    Args:
    - gas_data (dict): A dictionary containing gas dataframes.
    - gas_name (str): Name of the gas to filter (e.g., 'CO', 'NO2', 'O3', 'SO2').
    - start_idx (int): Starting index for filtering.
    - length (int): Length of data to filter.

    Returns:
    - pd.DataFrame: Filtered gas data for the specified gas and index range.
    """
    # Check if gas_name exists in gas_data
    if gas_name not in gas_data:
        raise ValueError(f"Gas '{gas_name}' not found in gas_data.")

    # Get the gas dataframe
    gas_df = gas_data[gas_name]

    # Filter data based on start_idx and length
    filtered_data = gas_df.iloc[start_idx:start_idx + length]

    return filtered_data


# In[ ]:
# 一体化函数
# All-in-one functions
def column_name_tolist(gas_name):
    #列名转换（可排除多余空格和其他字符）
    column_names = pd.Series(data.columns)
    filtered_columns = column_names[column_names.str.contains(gas_name)].tolist()
    Column_Name = filtered_columns[0]
    return Column_Name



# 列数据选择函数
# Column data selection function
def Data_integration(gas_name):
    column_name_tolist(gas_name)
    dataton = (data['Date'], data[column_name_tolist(gas_name)])
    return dataton



def Data_interception(gas_name,start_index,length):
    Date,gas_date= Data_integration(gas_name)
    Date_cut = Date[start_index:start_index+length:]
    gas_date_cut = gas_date[start_index:start_index+length:]
    return Date_cut, gas_date_cut

# In[ ]:
# 显示一周内每一小时对应的数据变化趋势
# Displays the trend of data changes for each hour of the week
def gas_curves_week(gas_name, ylabel, exportName, axis, label_no):
    axis.plot(Data_interception(gas_name,start_index=2841,length=168)[0], Data_interception(gas_name,start_index=2841,length=168)[1], alpha=0.7, label=gas_name, color=eval(gas_name+"Color"))
    axis.set_xlabel('Date',fontsize=16)
    axis.tick_params(axis='x', labelsize=16)
    #plt.xticks(rotation=45)  # 将横坐标值旋转45度显示
    axis.set_ylabel(ylabel)
    axis.text(0.05, 0.9, string.ascii_lowercase[label_no], transform=axis.transAxes, size=16, weight='bold')
    axis.legend()


fig, axw = plt.subplots(2,2,figsize=(12,10))
ax5 = axw[0, 0]
ax6 = axw[1, 0]
ax7 = axw[1, 1]
ax8 = axw[0, 1]

gas_curves_week('CO', 'CO (mg/m³)', 'exportName', ax5, 0)
gas_curves_week('NO2', 'NO2 (µg/m³)', 'exportName', ax7, 1)
gas_curves_week('O3', 'O3 (µg/m³)', 'exportName', ax6, 2)
gas_curves_week('SO2', 'SO2 (µg/m³)', 'exportName', ax8, 3)


fig.autofmt_xdate(rotation=90)
fig.align_ylabels(axs[:, 1])
plt.savefig('Sup_pics/Fig2.pdf', bbox_inches='tight')
plt.show()


# 以上为单显图
# The above is a single display
# 以下为合成同显图：区别主要在单位统一
# The following is the composite same display diagram: the difference is mainly in the unit

# 先进行数据处理
# Data processing is carried out first
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%dT%H:%M:%SZ")
data["CO (mg/m³)"] *= 1000

print(data["Date"])


# 显示在一张图上，由于显示对比度不够，不显示该图
# It is displayed on a single image, but the image is not displayed due to insufficient display contrast
"""def plot_trajectories_full(data, ylabel, exportName, axis,labelNo):
    #plot the quantity at all locations for the full year
    axis.plot(data["Date"],Data_integration("CO")[1], alpha=0.7, color=COColor)
    axis.plot(data["Date"],Data_integration("NO2")[1], alpha=0.7, color=NO2Color)
    axis.plot(data["Date"],Data_integration("O3")[1], alpha=0.7, color=O3Color)
    axis.plot(data["Date"],Data_integration("SO2")[1], alpha=0.7, color=SO2Color)
    plt.legend(['CO','NO2','O3','SO2'])
    axis.set_xlabel('Date')
    axis.set_ylabel(ylabel)
    axis.text(0.05, 0.9, string.ascii_lowercase[labelNo], transform=axis.transAxes,size=20, weight='bold')

fig, axall = plt.subplots(figsize=(12,10))
plot_trajectories_full(data, "Gas density (µg/m³)","exportName", axall, 1)
plt.savefig('Sup_pics/Fig3.pdf', bbox_inches='tight')
plt.show()
"""
# In[ ]:
# 绘制概率密度函数直方图
# Plot a histogram of the probability density function
def plot_histograms_ax(data,gas_name, ylabel, useCustomyLim, axis, labelNo):
    #plot the histograms at all locations for the same time
    current_palette = sns.color_palette()
    sns.distplot(Data_integration(gas_name)[1].dropna(), hist=True, kde=True, color=eval(gas_name+"Color"), ax=axis)
    #sns.histplot(Data_integration(gas_name)[1].dropna(), kde=True, color=eval(gas_name + "Color"), ax=axis)
    axis.set_yscale('log')
    axis.set_xlabel(ylabel)
    axis.set_ylabel('PDF')
    axis.text(0.05, 0.9, string.ascii_lowercase[labelNo], transform=axis.transAxes,size=20, weight='bold')
    if useCustomyLim:
        plt.ylim(10**-4,10**-0.5)

fig, axs = plt.subplots(2,2,figsize=(12,10))

ax9 = axs[0, 0]
ax10 = axs[1, 0]
ax11 = axs[1, 1]
ax12 = axs[0, 1]


plot_histograms_ax(data,'CO', 'CO (mg/m³)','EC', ax9, 0)
plot_histograms_ax(data,'NO2', 'NO2 (µg/m³)','EC', ax10, 0)
plot_histograms_ax(data,'O3', 'O3 (µg/m³)','EC', ax11, 0)
plot_histograms_ax(data,'SO2', 'SO2 (µg/m³)','EC', ax12, 0)

plt.savefig('Sup_pics/Fig4.pdf', bbox_inches='tight')
plt.show()


# In[ ]:

def seasonal_detrend(gas_name, frequency):
    hour=4
    day=24*hour
    #detrending
    gas_outdata = np.array(Data_integration(gas_name)[1].dropna())
    result_mul = seasonal_decompose(gas_outdata, model='multiplicative', extrapolate_trend='freq', period=frequency)
    detrended = gas_outdata - result_mul.trend
    return(detrended)



def emd_detrending(gas_name,numberOfOmmitedModes):
    #retrieve the station data to be analysed
    gas_outdata = np.array(Data_integration(gas_name)[1].dropna())
    #perform EMD analysis
    emd = EMD()
    IMFs = emd(gas_outdata)
    numberOfModes=len(IMFs)
    summedModes=np.zeros(len(IMFs[0]))
    for modeIndex in range(1,numberOfModes-numberOfOmmitedModes):
        #compute the new sum of modes that approximates the trend
        summedModes=summedModes+IMFs[numberOfModes-modeIndex]
    trend=summedModes
    return(gas_outdata-trend)
#function to select which de-trending method we apply
def detrending(gas_name, method, detrendingParameter):
    if method=='Seasonal':
        return seasonal_detrend(gas_name, detrendingParameter)
    if method=='EMD':
        return emd_detrending(gas_name, detrendingParameter)


def quantityLabels(gas_name):
    if gas_name=='CO':
        xlabel='CO concentration fluctuations [mg/m³]'
        exportName='CO'
    if gas_name=='NO2':
        xlabel='NO2 concentration fluctuations [µg/m³]'
        exportName='NO2'
    if gas_name=='SO2':
        xlabel='SO2 concentration fluctuations [µg/m³]'
        exportName='SO2'
    if gas_name=='O3':
        xlabel = 'O3 concentration fluctuations [µg/m³]'
        exportName = 'O3'
    return (xlabel,exportName)

# In[ ]:

def q_Gauss_pdf(x,q,l,mu):
    constant=np.sqrt(np.pi)*gamma((3-q)/(2*(q-1)))/(np.sqrt(q-1)*gamma(1/(q-1)))
    pdf=np.sqrt(l)/constant*(1+(1-q)*(-l*(x-mu)**2))**(1/(1-q))
    return pdf

class q_Gauss_custom(st.rv_continuous):
    def _pdf(self, x, q, l):
        "Custom q-Gauss distribution"
        #q=self.q
        #l=self.l
        mu=0
        pdf = q_Gauss_pdf(x,q,l,mu)
        return pdf
    def _stats(self, q, l):
        return [self.q,self.l,0,0]
    #fitstart provides a starting point for any MLE fit
    def _fitstart(self,data):
        return (1.1,1.1)
    def _argcheck(self, q,l):
        #define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
        largeQ = q > 1
        positiveScale=l>0
        all_bool = largeQ&positiveScale
        return all_bool
qGauss_custom_inst = q_Gauss_custom(name='qGauss_custom',a=0)

# In[ ]:

def plot_fluctuation_histo(detrended, gas_name, method):
    (xlabel, exportName) = quantityLabels(gas_name)
    # plt.hist(detrended, density=True,log=True)
    plot = sns.distplot(detrended)
    # extract distplot range
    (xvalues_hist, yvalues_hist) = plot.get_lines()[0].get_data()
    qGaussParameters = qGauss_custom_inst.fit(detrended - np.mean(detrended), 1.2, 1, floc=0, fscale=1)
    qGaussValues = q_Gauss_pdf(xvalues_hist, qGaussParameters[0], qGaussParameters[1], np.mean(detrended))
    plt.plot(xvalues_hist, qGaussValues)
    plt.legend(['q-Gaussian', 'Data'])
    plt.yscale('log')
    plt.ylim(0.1 * min(qGaussValues), 10 * max(qGaussValues))
    plt.xlabel(xlabel)
    plt.ylabel('PDF')
    plt.title(gas_name + ', ' + method + ', q=' + str(round(qGaussParameters[0], 3)))
    plt.savefig('Sup_pics/' + exportName + '_Fluctuation_' + method + '_Detrending_Hist' + gas_name + '.pdf',
                bbox_inches='tight')
    plt.show()

# 超统计函数
# Superstatistics functions

# 确定平均峰度
# Determine average kurtosis
def averageKappa(data, DeltaT):
    meanData = np.mean(data);
    tMax = len(data);
    nominator = sum((data[0:DeltaT] - meanData) ** 4)
    denominator = sum((data[0:DeltaT] - meanData) ** 2)
    sumOfFractions = nominator / (denominator ** 2);
    for i in range(DeltaT, tMax):
        nominator = nominator + (data[i] - meanData) ** 4 - (data[i - DeltaT] - meanData) ** 4;
        denominator = denominator + (data[i] - meanData) ** 2 - (data[i - DeltaT] - meanData) ** 2;
        sumOfFractions = sumOfFractions + nominator / (denominator ** 2);
    return sumOfFractions / (tMax - DeltaT) * DeltaT

# 定义一个隶属函数，避免包含已删除为 NaN 的索引
# define a membership function to avoid including indices that have been removed as NaN
def testMembership(item, list):
    if any(item == c for c in list):
        return True
    else:
        return False


def betaList(data, T):
    uSquareMean = sum(data[0:T] ** 2) / T;
    uMean = sum(data[0:T]) / T;
    betaValues = [1 / (uSquareMean - uMean ** 2)]
    tMax = len(data)
    for i in range(T + 1, tMax):
        uSquareMean = uSquareMean + (data[i] ** 2 - data[i - T] ** 2) / T
        uMean = uMean + (data[i] - data[i - T]) / T;
        betaValues.append(1 / (uSquareMean - uMean ** 2))
    return betaValues












# In[ ]:
hour = 4
# 计算并绘制平均峰度随时间的变化
# compute and plot the average kurtosis as a function of time
def plotLongTimeScale(detrended, gas_name, method, startTime, EndTime, TimeStep, targetKurtosis):
    (xlabel, exportName) = quantityLabels(gas_name)
    kurtosisList = []
    timeList = range(startTime * hour, EndTime * hour, TimeStep * hour)
    plotTimeList = range(startTime, EndTime, TimeStep)
    for time in timeList:
        kurtosisList.append(averageKappa(detrended, time))
    plt.plot(plotTimeList, kurtosisList, linewidth=4.0)
    plt.plot(plotTimeList, targetKurtosis * np.ones(len(kurtosisList)), linewidth=4.0)
    plt.xlabel('Time lag $\Delta t$ [hour]')
    plt.ylabel('Average kurtosis $\overline{\kappa}$')
    plt.title(gas_name)
    plt.legend([gas_name + ' ' + exportName, 'Gaussian'])
    plt.savefig('Sup_pics/' + 'Long_time_scales_' + exportName + '_' + method + '_Detrending_' + gas_name + '.pdf',
                bbox_inches='tight')
    plt.show()

# 绘制低方差和高方差快照
# plot a low and high-variance snapshot
def plotExtremeSnapshots(detrended, gas_name, method, longTimeScale):
    (xlabel, exportName) = quantityLabels(gas_name)
    longTimeScaleApplied = round(longTimeScale)
    # compute the variance of the different time windows
    varianceList = []
    for tIndex in range(0, int(len(detrended) / longTimeScaleApplied)):
        varianceList.append(np.std(detrended[tIndex * longTimeScaleApplied:(tIndex + 1) * longTimeScaleApplied]))
    minposition = [i for i, x in enumerate(varianceList) if x == min(varianceList)]
    maxposition = [i for i, x in enumerate(varianceList) if x == max(varianceList)]
    # display two different histograms using only data within one "homogeneous" time window
    startingIndex1 = minposition[0]
    startingIndex2 = maxposition[0]
    # set up subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # plot log-scale plot
    sns.distplot(detrended[startingIndex1 * longTimeScaleApplied:(startingIndex1 + 1) * longTimeScaleApplied], ax=ax1)
    ax1.set_yscale('log')
    # ax1.hist(detrended[startingIndex1*longTimeScaleApplied:(startingIndex1+1)*longTimeScaleApplied], density=True,log=True,bins=10)
    ax1.set_ylabel('PDF');
    ax1.set_xlabel(xlabel);
    ax1.set_title(gas_name + ', ' + method + ', Low variance')
    ax1.text(0.05, 0.9, string.ascii_lowercase[0], transform=ax1.transAxes,
             size=20, weight='bold')
    # plot linear-scale plot
    # ax2.hist(detrended[startingIndex2*longTimeScaleApplied:(startingIndex2+1)*longTimeScaleApplied], density=True,log=True,bins=10)
    sns.distplot(detrended[startingIndex2 * longTimeScaleApplied:(startingIndex2 + 1) * longTimeScaleApplied], ax=ax2)
    ax2.set_yscale('log')
    ax2.set_ylabel('PDF');
    ax2.set_xlabel(xlabel);
    ax2.set_title(gas_name + ', ' + method + ', High variance')
    ax2.text(0.05, 0.9, string.ascii_lowercase[1], transform=ax2.transAxes,
             size=20, weight='bold')
    # combine and show both plots
    f.subplots_adjust(wspace=0.2)
    plt.savefig('Sup_pics/' + 'ExtremeSnapshots_' + exportName + '_' + method + '_Detrending_' + gas_name + '.pdf',
                bbox_inches='tight')
    plt.show()



# In[ ]:
# 返回下一个预测函数，假设峰度是线性函数
# function that returns the next guess, assuming the kurtosis is a linear function
def nextGuessingTime(x1, y1, x2, y2, targetKurtosis):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    predictedTime = (targetKurtosis - intercept) / slope
    return int(predictedTime)

# 系统地确定长时间尺度的功能，给定初始值和一定的容差
# Function to systemnatically determine the long time scale, given an initial value and a certain tolerance
def determineLongTimeScale(detrended, gas_name, method, initialTimeGuess, initalGuessIncrement,
                           kurtosisTolerance, targetKurtosis):
    (xlabel, exportName) = quantityLabels(gas_name)
    kurtosisList = []
    timeList = []
    # we simplify the taks by assuming that the long time scale is approxiamtely linear with delta t
    # inital 2 runs:
    averageKurtosis = averageKappa(detrended, initialTimeGuess)
    kurtosisList.append(averageKurtosis)
    timeList.append(initialTimeGuess)
    if averageKurtosis > targetKurtosis:
        newTime = initialTimeGuess - initalGuessIncrement
    else:
        newTime = initialTimeGuess + initalGuessIncrement
    timeList.append(newTime)
    averageKurtosis = averageKappa(detrended, newTime)
    kurtosisList.append(averageKurtosis)
    # initiate a counter to prevent endless loops
    loopCounter = 0
    while abs(averageKurtosis - targetKurtosis) > kurtosisTolerance:
        # only repeat the loop 100 times
        if loopCounter == 100:
            return 'No convergence'
        newTime = nextGuessingTime(timeList[-1], kurtosisList[-1], timeList[-2], kurtosisList[-2], targetKurtosis)
        if newTime < 0:
            return 'No convergence'
        timeList.append(newTime)
        averageKurtosis = averageKappa(detrended, newTime)
        # abort if average kurtosis becomes negative
        if averageKurtosis < 0:
            return 'No convergence'
        kurtosisList.append(averageKurtosis)
        loopCounter = loopCounter + 1
        # print(loopCounter)
        # print(averageKurtosis)
    return (timeList[-1], 'Converged to kurtosis=' + str(averageKurtosis), 'Iterations needed=' + str(loopCounter))


def fit_and_plot_betaDist(detrended, gas_name, method, longTimeScale):
    (xlabel, exportName) = quantityLabels(gas_name)
    betaDis = betaList(detrended, round(longTimeScale))
    # some filtering to avoid negative or highly positive values, skewing the figures
    betaDis = np.array(betaDis)
    betaDis = betaDis[betaDis < np.quantile(betaDis, 0.99)]
    betaDis = betaDis[betaDis >= 0]
    meanBeta = np.mean(betaDis)
    meanB = meanBeta

    # 定义自定义 PDF，使用 beta 的平均值进行逆 chi^2 分布，但将其保留为 chi^2 分布的拟合参数（导致更稳定的拟合）
    # 定义卡方自定义 PDF，注意我们还拟合了平均 beta 值以允许算法收敛
    # 定义自定义 PDF
    # define custom pdf, using the mean of beta for inverse chi^2 distributions but leaving it as a fitting parameter for chi^2 distributions (leads to stabler fits)
    # define the chi-square custom pdf, note that we also fit the mean beta value to allow the algorithm to converge
    # define custom pdf
    class chiSquare_custom(st.rv_continuous):
        def _pdf(self, x, degreesN, meanB):
            "Custom Chi-square distribution"
            pdf = 1 / (gamma(degreesN / 2)) * (degreesN / (2 * meanB)) ** (degreesN / 2) * x ** (
                        degreesN / 2 - 1) * np.exp(-(degreesN * x) / (2 * meanB))
            return pdf

        def _stats(self, degreesN, meanB):
            return [self.degreesN, self.meanB, 0, 0]

        # fitstart provides a starting point for any MLE fit
        def _fitstart(self, data):
            return (1.1, 1.1)

        def _argcheck(self, degreesN, meanB):
            # define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees = degreesN > 0
            positiveMean = meanB > 0
            all_bool = positiveDegrees & positiveMean
            return all_bool

    class invChiSquare_custom(st.rv_continuous):
        def _pdf(self, x, degreesN):
            "Custom inverse Chi-square distribution"
            pdf = 1 / (gamma(degreesN / 2)) * meanB * (degreesN * meanB / 2) ** (degreesN / 2) * x ** (
                        -degreesN / 2 - 2) * np.exp(-(degreesN * meanB) / (2 * x))
            return pdf

        def _stats(self, degreesN):
            return [self.degreesN, 0, 0]

        # fitstart provides a starting point for any MLE fit
        def _fitstart(self, data):
            return (1.1, 1.1)

        def _argcheck(self, degreesN):
            # define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees = degreesN > 0
            all_bool = positiveDegrees
            return all_bool

    # initiate distribution instances with support starting at 0
    chiSquare_custom_inst = chiSquare_custom(name='chiSquare_custom', a=0)
    invchiSquare_custom_inst = invChiSquare_custom(name='invchiSquare_custom', a=0)

    logNormalFit = stats.lognorm.fit(betaDis, scale=np.exp(5), loc=0)
    chiSquareFit = chiSquare_custom_inst.fit(betaDis, 3, meanBeta, floc=0, fscale=1)
    degrees_chi = chiSquareFit[0]
    inv_chiSquareFit = invchiSquare_custom_inst.fit(betaDis, 1, floc=0, fscale=1)
    degrees_inv_chi = inv_chiSquareFit[0]
    # Compare the fits with the data:
    xrange = np.arange(0, max(betaDis), max(betaDis) / 100)
    pdfVlaues_chi = chiSquare_custom_inst.pdf(xrange, degreesN=degrees_chi, meanB=meanBeta, loc=0, scale=1)
    pdfVlaues_inv_chi = invchiSquare_custom_inst.pdf(xrange, degreesN=degrees_inv_chi, loc=0, scale=1)
    pdfVlaues_logNorm = stats.lognorm.pdf(xrange, *logNormalFit)
    plot = sns.distplot(betaDis)
    # plt.hist(betaDis, density=True)
    plt.plot(xrange, pdfVlaues_chi, linewidth=4.0)
    plt.plot(xrange, pdfVlaues_logNorm, linewidth=4.0)
    plt.plot(xrange, pdfVlaues_inv_chi, linewidth=4.0)
    # plt.yscale('log')
    plt.title(gas_name + ', T=' + str(round(longTimeScale)) + 'h, $n_{\chi^2}$= ' + str(
        round(degrees_chi, 3)) + ', $n_{inv. \chi^2}$= ' + str(round(degrees_inv_chi, 3)))
    plt.xlabel(r'$ \beta $ (' + exportName + ')')
    plt.ylabel("PDF")
    # extract distplot range
    (xvalues_hist, yvalues_hist) = plot.get_lines()[0].get_data()
    plt.ylim(min(yvalues_hist), max(yvalues_hist) * 1.1)
    plt.xlim(0)
    plt.legend(['$\chi^2$', 'log-norm.', 'inv. $\chi^2$', r'$\beta$ values'])
    plt.savefig('Sup_pics/' + 'BetaDistribution_' + exportName + '_' + method + '_Detrending_' + gas_name + '.pdf',
                bbox_inches='tight')
    plt.show()


# In[ ]:

# BETA 混合分布和拟合
# BETA MIXTURE PLOT AND FIT
def fit_and_plot_betaDist_Mix(detrended, gas_name, method, longTimeScale):
    (xlabel, exportName) = quantityLabels(gas_name)
    betaDis = betaList(detrended, round(longTimeScale))
    # some filtering to avoid negative or highly positive values, skewing the figures
    betaDis = np.array(betaDis)
    betaDis = betaDis[betaDis < np.quantile(betaDis, 0.99)]
    betaDis = betaDis[betaDis >= 0]
    meanBeta = np.mean(betaDis)
    meanB = meanBeta

    # 定义自定义 PDF，使用 beta 的平均值进行逆 chi^2 分布，但将其保留为 chi^2 分布的拟合参数（导致更稳定的拟合）
    # 定义卡方自定义 PDF，注意我们还拟合了平均 beta 值以允许算法收敛
    # 定义自定义 PDF
    # define custom pdf, using the mean of beta for inverse chi^2 distributions but leaving it as a fitting parameter for chi^2 distributions (leads to stabler fits)
    # define the chi-square custom pdf, note that we also fit the mean beta value to allow the algorithm to converge
    # define custom pdf

    def chiSquarePDF(x, degreesN, meanB):
        return 1 / (gamma(degreesN / 2)) * (degreesN / (2 * meanB)) ** (degreesN / 2) * x ** (
                    degreesN / 2 - 1) * np.exp(-(degreesN * x) / (2 * meanB))

    class chiSquare_custom(st.rv_continuous):
        def _pdf(self, x, degreesN, meanB):
            "Custom Chi-square distribution"
            pdf = chiSquarePDF(x, degreesN, meanB)
            return pdf

        def _stats(self, degreesN, meanB):
            return [self.degreesN, self.meanB, 0, 0]

        # fitstart provides a starting point for any MLE fit
        def _fitstart(self, data):
            return (1.1, 1.1)

        def _argcheck(self, degreesN, meanB):
            # define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees = degreesN > 0
            positiveMean = meanB > 0
            all_bool = positiveDegrees & positiveMean
            return all_bool

    class chiSquare_custom_mix(st.rv_continuous):
        def _pdf(self, x, degreesN1, degreesN2, weightW, meanB):
            "Custom Chi-square distribution"
            pdf = weightW * chiSquarePDF(x, degreesN1, meanB) + (1 - weightW) * chiSquarePDF(x, degreesN2, meanB)
            return pdf

        def _stats(self, degreesN1, degreesN2, weightW, meanB):
            return [self.degreesN1, self.degreesN2, self.weightW, self.meanB, 0, 0]

        # fitstart provides a starting point for any MLE fit
        def _fitstart(self, data):
            return (1.1, 1.1, 0.5, 1.1)

        def _argcheck(self, degreesN1, degreesN2, weightW, meanB):
            # define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees1 = degreesN1 > 0
            positiveDegrees2 = degreesN2 > 0
            positiveMean = meanB > 0
            positiveWeight = weightW > 0
            maxWeight = weightW < 1
            all_bool = positiveDegrees1 & positiveDegrees2 & positiveMean & positiveWeight & maxWeight
            return all_bool

    # initiate distribution instances with support starting at 0
    chiSquare_custom_mix_inst = chiSquare_custom_mix(name='chiSquare_mix_custom', a=0)
    chiSquare_custom_inst = chiSquare_custom(name='chiSquare_custom', a=0)

    chiSquareFit_single = chiSquare_custom_inst.fit(betaDis, 3, meanBeta, floc=0, fscale=1)
    degrees_chi = chiSquareFit_single[0]
    chiSquareFit_mix = chiSquare_custom_mix_inst.fit(betaDis, 3, 3, 0.5, meanBeta, floc=0, fscale=1)
    degrees_chi1 = chiSquareFit_mix[0]
    degrees_chi2 = chiSquareFit_mix[1]
    weight_W = chiSquareFit_mix[2]
    fitMean_B = chiSquareFit_mix[3]
    # print('Fit beta='+str(fitMean_B))
    # print('Data beta='+str(meanBeta))
    # Compare the fits with the data:
    xrange = np.arange(0, max(betaDis), max(betaDis) / 100)
    pdfVlaues_chi_mix = chiSquare_custom_mix_inst.pdf(xrange, degreesN1=degrees_chi1, degreesN2=degrees_chi2,
                                                      weightW=weight_W, meanB=fitMean_B, loc=0, scale=1)
    pdfVlaues_chi_single = chiSquare_custom_inst.pdf(xrange, degreesN=degrees_chi, meanB=meanBeta, loc=0, scale=1)
    plot = sns.distplot(betaDis)
    # plt.hist(betaDis, density=True)
    plt.plot(xrange, pdfVlaues_chi_single, linewidth=4.0, color='tab:gray', linestyle='--')
    plt.plot(xrange, pdfVlaues_chi_mix, linewidth=4.0, color='black')
    # plt.yscale('log')
    plt.title(gas_name + ', T=' + str(round(longTimeScale)) + r'h, $n_{\chi_1}$=' + str(
        round(degrees_chi1, 2)) + ', $n_{\chi_2}$=' + str(round(degrees_chi2, 2)) + ', $W$=' + str(round(weight_W, 2)))
    plt.xlabel(r'$ \beta $ (' + exportName + ')')
    plt.ylabel("PDF")
    # extract distplot range
    (xvalues_hist, yvalues_hist) = plot.get_lines()[0].get_data()
    plt.ylim(min(yvalues_hist), max(yvalues_hist) * 1.1)
    plt.xlim(0)
    plt.legend(['Single. $\chi^2$', 'Mix. $\chi^2$', r'$\beta$ values'])
    plt.savefig(
        'Sup_pics/' + 'BetaDistribution_' + exportName + '_' + method + '_Detrending_MIXTURE_' + gas_name + '.pdf',
        bbox_inches='tight')
    plt.show()






# In[ ]:
# 需要将时间序列数据转换为字符串数据用于处理
# Time series data needs to be converted to string data for processing
def save_txt_fig(gas_name, fq = 6, IMF = 2):
    dataClean = data["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')

    #print(dataClean)

    hour = 4
    #gas_name = 'CO'
    #dates = data["Date"] #数据格式不正确

    dates = dataClean
    #dates = dates.astype('object')
    #print(dates)


    stdata = Data_integration(gas_name)[1].dropna()

    #save txt with method = 'Seasonal'
    method = 'Seasonal'  # 'EMD', 'Seasonal'
    detrendingParameter = fq * hour  # 6*hour 2
    fluctuations_Seasonal_1 = detrending(gas_name, method, detrendingParameter)
    detrendingParameter = 12 * hour  # 6*hour 2
    fluctuations_Seasonal_2 = detrending(gas_name, method, detrendingParameter)
    trend_Seasonal_1 = stdata - fluctuations_Seasonal_1
    trend_Seasonal_2 = stdata - fluctuations_Seasonal_2
    np.savetxt("data/Detrended" + str(gas_name) + str(method) + ".csv", fluctuations_Seasonal_1, delimiter=",")
    #np.savetxt("data/Detrended" + str(gas_name) + str(method) + ".csv", fluctuations_Seasonal_1, delimiter=",")







    method = 'EMD'  # 'EMD', 'Seasonal'
    detrendingParameter = IMF  # 6*hour 2
    fluctuations_EMD_1 = detrending(gas_name, method, detrendingParameter)
    detrendingParameter = 3  # 6*hour 2
    fluctuations_EMD_2 = detrending(gas_name, method, detrendingParameter)
    trend_EMD_1 = stdata - fluctuations_EMD_1
    trend_EMD_2 = stdata - fluctuations_EMD_2

    #save txt:
    np.savetxt("data/Detrended" + str(gas_name) + str(method) + ".csv", fluctuations_EMD_1, delimiter=",")



    plt.rc('font', size=14)
    (xlabel, exportName) = quantityLabels(gas_name)
    offSet = 2 + 4 * 9 + 12 * 4
    length = hour * 24 * 7
    appliedDates = dates.tolist()[offSet:offSet + length]
    #print(appliedDates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(appliedDates, stdata[offSet:offSet + length])
    ax1.plot(appliedDates, trend_Seasonal_1[offSet:offSet + length])
    ax1.plot(appliedDates, trend_Seasonal_2[offSet:offSet + length])
    ax1.legend(['Data', "Seasonal trend (f="+str(fq)+"h)", 'Seasonal trend (f=12h)'], loc=4)
    ax2.plot(appliedDates, stdata[offSet:offSet + length])
    ax2.plot(appliedDates, trend_EMD_1[offSet:offSet + length])
    ax2.plot(appliedDates, trend_EMD_2[offSet:offSet + length])
    ax2.legend(['Data', "EMD trend (m="+str(IMF)+")", 'EMD trend (m=3)'], loc=4)
    ax1.text(0.05, 0.9, string.ascii_lowercase[0], transform=ax1.transAxes, size=20, weight='bold')
    ax2.text(0.05, 0.9, string.ascii_lowercase[1], transform=ax2.transAxes, size=20, weight='bold')
    for axs in (ax1, ax2):
        # axs.set_xticks([appliedDates[0],appliedDates[2*24*4],appliedDates[4*24*4],appliedDates[6*24*4]],[appliedDates[0][0:10],appliedDates[2*24*4][0:10],appliedDates[4*24*4][0:10],appliedDates[6*24*4][0:10]])
        axs.set_xticks([appliedDates[0], appliedDates[2 * 24 * 4], appliedDates[4 * 24 * 4], appliedDates[6 * 24 * 4]])
        axs.set_xticklabels([appliedDates[0][5:10], appliedDates[2 * 24 * 4][5:10], appliedDates[4 * 24 * 4][5:10],
                             appliedDates[6 * 24 * 4][5:10]])
        # axs.set_xticks([appliedDates[0],appliedDates[2*24*4],appliedDates[4*24*4],appliedDates[6*24*4]])
        axs.tick_params(axis='x', labelrotation=90)
        axs.set_ylabel(xlabel)
        axs.set_xlabel('Date')
    fig.subplots_adjust(wspace=0.4)
    plt.savefig('Sup_pics/Fig_'+str(gas_name) + 'detrended' +'.pdf', bbox_inches='tight')
    plt.show()



save_txt_fig("CO",3,1)
save_txt_fig("NO2",3,1)
save_txt_fig("SO2",3,1)
save_txt_fig("O3",3,1)








# In[ ]:

def import_detrending(gas_name, method):
    detrended = np.loadtxt("data/Detrended" + str(gas_name) + str(method) + ".csv", delimiter=",")
    return detrended


def plot_fluctuation_histo_axs(gas_name, method, axis, labelNo):
    #print(quantityLabels(gas_name))
    (xlabel, exportName) = quantityLabels(gas_name)
    # plt.hist(detrended, density=True,log=True)
    detrended = import_detrending(gas_name, method)
    plot = sns.distplot(detrended, ax=axis)
    #plot = sns.histplot(detrended, ax=axis)
    # extract distplot range
    (xvalues_hist, yvalues_hist) = plot.get_lines()[0].get_data()
    qGaussParameters = qGauss_custom_inst.fit(detrended - np.mean(detrended), 1.2, 1, floc=0, fscale=1)
    qGaussValues = q_Gauss_pdf(xvalues_hist, qGaussParameters[0], qGaussParameters[1], np.mean(detrended))
    axis.plot(xvalues_hist, qGaussValues)
    plt.legend(['q-Gaussian', 'Data'], loc=4)
    axis.set_yscale('log')
    axis.set_ylim(0.1 * min(qGaussValues), 10 * max(qGaussValues))
    axis.set_xlabel(xlabel)
    axis.set_ylabel('PDF')
    axis.set_title(gas_name + ', ' + method + ', q=' + str(round(qGaussParameters[0], 3)))
    axis.text(0.05, 0.9, string.ascii_lowercase[labelNo], transform=axis.transAxes, size=20, weight='bold')


# set up subplots
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
plot_fluctuation_histo_axs('CO', 'Seasonal', ax1, 0)
plot_fluctuation_histo_axs('NO2', 'Seasonal', ax2, 1)
plot_fluctuation_histo_axs('SO2', 'Seasonal', ax3, 2)
plot_fluctuation_histo_axs('O3', 'Seasonal', ax4, 3)
fig1.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('Sup_pics/Fig5.pdf', bbox_inches='tight')
plt.show()

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
plot_fluctuation_histo_axs('CO', 'EMD', ax1, 0)
plot_fluctuation_histo_axs('NO2', 'EMD', ax2, 1)
plot_fluctuation_histo_axs('SO2', 'EMD', ax3, 2)
plot_fluctuation_histo_axs('O3', 'EMD', ax4, 3)
fig2.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('Sup_pics/Fig6.pdf', bbox_inches='tight')
plt.show()















def import_detrending(gas_name, method):
    detrended = np.loadtxt("data/Detrended" + str(gas_name) + str(method) + ".csv", delimiter=",")
    return detrended


def gas_out(gas_name):
    hour = 4
    method = 'Seasonal'
    exportName = 'XU ZHOU'
    startTime = 2
    EndTime = 20
    TimeStep = 2
    targetKurtosis = 3
    detrended = import_detrending(gas_name, method)
    kurtosisList = []
    timeList = range(startTime * hour, EndTime * hour, TimeStep * hour)
    plotTimeList = range(startTime, EndTime, TimeStep)
    for time in timeList:
        kurtosisList.append(averageKappa(detrended, time))
    plt.plot(plotTimeList, kurtosisList, linewidth=4.0)
    plt.plot(plotTimeList, targetKurtosis * np.ones(len(kurtosisList)), linewidth=4.0)
    plt.xlabel('Time lag $\Delta t$ [hour]')
    plt.ylabel('Average kurtosis $\overline{\kappa}$')
    plt.title(column_name_tolist(gas_name))
    plt.legend([column_name_tolist(gas_name) + ' - ' + exportName, 'Gaussian'])
    plt.savefig('Sup_pics/Fig'+gas_name+'Gaussian' +'.pdf', bbox_inches='tight')
    plt.show()




gas_out("CO")
gas_out("NO2")
gas_out("SO2")
gas_out("O3")

