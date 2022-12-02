import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd


x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)



 #Mean
'''
The sample mean, also called the sample arithmetic mean or simply the average, is 
the arithmetic average of all the items in a dataset. The mean of a dataset 𝑥 is 
mathematically expressed as Σᵢ𝑥ᵢ/𝑛, where 𝑖 = 1, 2, …, 𝑛. In other words, it’s the sum of all the elements 𝑥ᵢ 
divided by the number of items in the dataset 𝑥. '''

mean_ = statistics.mean(x)
meanf_ = statistics.fmean(x)
meanar_ = y.mean()
meanwithnan = np.nanmean(y_with_nan)
meanserias_ = z.mean()
meanseriaswithnan_= z_with_nan.mean()

#Weighted Mean

''' The weighted mean, also called the weighted arithmetic mean or weighted average, 
is a generalization of the arithmetic mean that enables you to define the relative contribution of each data point to the result.
You define one weight 𝑤ᵢ for each data point 𝑥ᵢ of the dataset 𝑥, where 𝑖 = 1, 2, …, 𝑛 and 𝑛 is the number of items in 𝑥. 
Then, you multiply each data point with the corresponding weight, sum all the products, 
and divide the obtained sum with the sum of weights: Σᵢ(𝑤ᵢ𝑥ᵢ) / Σᵢ𝑤ᵢ. '''

y, z, w = np.array(x), pd.Series(x), np.array(w)

wmean = np.average(y, weights=w)

#Geometric Mean
''' The geometric mean is the 𝑛-th root of the product of all 𝑛 elements 𝑥ᵢ in a dataset 𝑥: ⁿ√(Πᵢ𝑥ᵢ), 
where 𝑖 = 1, 2, …, 𝑛. 
 '''

gmean = statistics.geometric_mean(x)

# Median
''' The sample median is the middle element of a sorted dataset. The dataset can be sorted in increasing or decreasing order.
If the number of elements 𝑛 of the dataset is odd, then the median is the value at the middle position: 0.5(𝑛 + 1). 
If 𝑛 is even, then the median is the arithmetic mean of the two values in the middle, that is, the items at the positions 0.5𝑛 
and 0.5𝑛 + 1. '''

median_ = statistics.median(x)
median_1 = statistics.median(x[:-1])
low = statistics.median_low(x[:-1])
hight = statistics.median_high(x[:-1])
medianWithNan = np.nanmedian(y_with_nan)

#Mode
''' The sample mode is the value in the dataset that occurs most frequently. If there isn’t a single such value, 
then the set is multimodal since it has multiple modal values. For example, in the set that contains the points 2, 3, 2, 8, and 12,
the number 2 is the mode because it occurs twice, unlike the other items that occur only once. '''

u = [2, 3, 2, 8, 12]
v = [12, 15, 12, 15, 21, 15, 12]
mode_ = statistics.mode(u)
multimode_= statistics.multimode(v)


#Variance
''' The sample variance quantifies the spread of the data. It shows numerically how far the data points are from the mean.
You can express the sample variance of the dataset 𝑥 with 𝑛 elements mathematically as 𝑠² = Σᵢ(𝑥ᵢ − mean(𝑥))² / (𝑛 − 1), 
where 𝑖 = 1, 2, …, 𝑛 and mean(𝑥) is the sample mean of 𝑥. ''' 

''' You calculate the population variance similarly to the sample variance. However, you have to use 𝑛 in 
the denominator instead of 𝑛 − 1: Σᵢ(𝑥ᵢ − mean(𝑥))² / 𝑛. In this case, 𝑛 is the number of items in the entire population. 
You can get the population variance similar to the sample variance, with the following differences:

Replace (n - 1) with n in the pure Python implementation.
Use statistics.pvariance() instead of statistics.variance().
Specify the parameter ddof=0 if you use NumPy or Pandas. In NumPy, you can omit ddof because its default value is 0. '''


''' var_ = statistics.variance(y)
VarWithNan = np.nanvar(y_with_nan, ddof=1)
Var_dot = z.var(ddof=1)
Var_dotWithNan = z_with_nan.var(ddof=1) '''

# Standard Deviation
''' The sample standard deviation is another measure of data spread. It’s connected to the sample variance, as 
standard deviation, 𝑠, is the positive square root of the sample variance. '''

''' std_ = statistics.stdev(x)
std_with_nan = np.nanstd(y_with_nan, ddof=1) '''

# Percentiles

# x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
# Yperc = np.percentile(x, 95)

#Ranges 
''' x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
XRange = np.ptp(x) '''

# Measures of Correlation Between Pairs of Data

''' x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)
cov_matrix = np.cov(x_, y_)
var_ = statistics.variance(x)
r, p = scipy.stats.pearsonr(x_, y_)
corr_matrix = np.corrcoef(x_, y_)
LinReg  = scipy.stats.linregress(x_, y_)
rLinReg = LinReg.rvalue 
rPandaSeria = x__.corr(y__)
print(r, p)
print(corr_matrix)
print(rLinReg)
print(rPandaSeria) '''

# Working With 2D Data
    # Axes
''' a = np.array([[1, 1, 1],[2, 3, 1],[4, 9, 2],[8, 27, 4],[16, 1, 1]])
meanAxis0 = np.mean(a, axis=1)
describeA = scipy.stats.describe(a, axis=None, ddof=1, bias=False)
describeAdefault = scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0 '''

# DataFrames
''' a = np.array([[1, 1, 1],[2, 3, 1],[4, 9, 2],[8, 27, 4],[16, 1, 1]])
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
DFMean = df.mean()
DFMeanAxis = df.mean(axis=1)
print(df,DFMean,DFMeanAxis) '''

# Visualizing Data

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Box Plots
''' np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10) '''

''' fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show() '''

#Histograms

''' hist, bin_edges = np.histogram(x, bins=10)
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show() '''

# Histogram with the cumulative numbers of items

''' hist, bin_edges = np.histogram(x, bins=10)
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show() '''

# Pie Charts

''' x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show() '''

# Bar Charts

''' x = np.arange(21)
y = np.random.randint(21, size=21)
fig, ax = plt.subplots()
ax.bar(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show() '''

# X-Y Plots with regression
''' x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show() '''

# Heatmaps
''' x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show() '''

# hypergeometric probability distribution
''' from scipy.stats import hypergeom
import matplotlib.pyplot as plt
[M, n, N] = [500, 5, 50]
# 100 - общая выборка, 5 число бракованных единиц, 10 - случайная выборка
rv = hypergeom(M, n, N)
x = np.arange(0, n+1)
pmf_dogs =rv.cdf(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, pmf_dogs, 'bo')
ax.vlines(x, 0, pmf_dogs, lw=2)
ax.set_xlabel('Кол-во бракованных единиц в выборке')
ax.set_ylabel('hypergeom PMF')
plt.show() '''

# OC curve
from scipy.stats import binom 
from scipy.stats import hypergeom
p = [0.0005,0.005,0.008,0.01,0.02,.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
n = 50
Pa = []
PgInf = []
d= 2
N=1000
Alfa = 0.9
DeltaAlfa = 0.005
beta = 0.10
DeltaBeta = 0.005
LTPD = 0.03
Start_n_ForSearch = N / 10
End_n_ForSearch = N / 2
SolutionBeta = []
SolutionAlfa = []
# If we have N<300 it's better to use hypergeom distribution

''' # M - общая выборка, dg число бракованных единиц, n - случайная выборка
rv = hypergeom(M, dg, N)
x = np.arange(0, n+1)
pmf_dogs =rv.pmf(x) '''


for i in p:
        binomP = 0
        hypergP = 0 
        for iter in range(d+1):
            binomP = binomP + binom.pmf(iter,n, i)
            defective = N*i
            rv = hypergeom(N, defective, n)
            hypergP=hypergP+rv.pmf(iter)
        Pa.append(binomP) 
        PgInf.append(hypergP) 
        


fig, ax = plt.subplots()
ax.plot(p, Pa, linewidth=0, marker='s', label='Data points')
LabelString = 'Binomial: '+'n='+str(n) + ', c ='+ str(d)
LabelStringInf = 'Hypergeometric: '+'M='+str(N) + ', c ='+ str(d)
ax.plot(p, Pa, label=LabelString, color = 'green')
ax.plot(p, PgInf, label=LabelStringInf, color = 'red')
ax.set_xlabel('p')
ax.set_ylabel('Pa')
ax.legend(facecolor='white')
plt.show()


if (N/2) < Start_n_ForSearch:
    print("Общий объем партии N/2 {}, должен быть больше стартового значение выборки Start_n_ForSearch {}, поиск выборочного значения для бета риска покупателя {} не производился".format(End_n_ForSearch,Start_n_ForSearch,beta))
else:
    for iter_n in range(int(Start_n_ForSearch),int(End_n_ForSearch)):
        if N < 300:
            
            hypergP = 0
            
            for iter in range(d+1):
                defective = N*LTPD
                rv = hypergeom(N, defective, iter_n)
                hypergP=hypergP+rv.pmf(iter)
            if  (hypergP - beta) < DeltaBeta:
                CurrentSolution = (iter_n,hypergP,iter)
                SolutionBeta.append(CurrentSolution)
        else:
            
            binomP = 0
           
            for iter in range(d+1):
                binomP = binomP + binom.pmf(iter,iter_n, LTPD)
            if  (binomP - beta) < DeltaBeta:
                CurrentSolution = (iter_n,binomP,iter)
                SolutionBeta.append(CurrentSolution)
                break
        #if len(SolutionBeta) > 0:
        #   break
                
if len(SolutionBeta) > 0:                 
    
    print("Найдено решение бетта вероятности для заданных условий: ",SolutionBeta.sort())
    IterP = 0.001
    
    
    for SolAlfa in SolutionBeta:
        sample = SolAlfa[0]
        
        if N < 300:
                
            hypergP = 0
            
            for iter in range(d+1):
                defective = N*IterP
                rv = hypergeom(N, defective, sample[0])
                hypergP=hypergP+rv.pmf(iter)
            if  abs(hypergP - Alfa) < DeltaAlfa:
                CurrentSolution = (sample[0],hypergP,IterP,iter)
                SolutionAlfa.append(CurrentSolution)
        else:
            
                binomP = 0
            
                for iter in range(d+1):
                    binomP = binomP + binom.pmf(iter,sample, IterP)
                if  abs(binomP - Alfa) < DeltaAlfa:
                    CurrentSolution = (IterP,sample,binomP,iter)
                    SolutionAlfa.append(CurrentSolution)
                    break  
        if IterP < 1:
                IterP += 0.0001
        else:
                break
            
    if len(SolutionAlfa) > 0:                 
        print("Найдено решение альфа вероятности для заданных условий: ",SolutionAlfa.sort())
    else:
        print("Не найдено решение альфа вероятности  для заданных условий") 
        
    
else:
    print("Не найдено решение бетта вероятности  для заданных условий")        
            
                


