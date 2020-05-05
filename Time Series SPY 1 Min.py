#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


# In[2]:


df = pd.read_csv("SPY1Min.csv")


# In[3]:


df1 = df.copy()


# In[4]:


df1.head(10)


# In[5]:


df1.describe()


# In[6]:


df1.isna()


# In[7]:


df1.isna().sum()


# In[8]:


df1.marketClose.plot(figsize = (20,5), title = "SPY INTRADAY")
plt.show()


# In[9]:


# Time Series(stock price) data does not follow normal distribution because it does not follow Gauss-Markov assumption


# In[10]:


import scipy.stats
import pylab


# ### X axis - Theoritical quantiles is how many standard deviaiton away from mean
# ### Y axis - Stock Prices
# 

# In[11]:


# Quantile-Quantile plot(QQ plot) is used to check whether the distribution is normal or not


# In[12]:


scipy.stats.probplot(df1.marketClose,plot = pylab)
pylab.show()


# In[13]:


###Not Normally distributed as it is not following linear graph


# In[14]:


#df1["DateTime"] =  df1['Date'] + ' ' + df1['Time']


# In[15]:


df1.head(10)


# In[16]:


df1['DateTime'] = pd.to_datetime(df1.DateTime,dayfirst = True)


# In[17]:


df1.head(10)


# In[18]:


df1.DateTime.describe()


# In[19]:


#df1 = df1.drop(['Date','Time'],axis=1)


# In[20]:


#df1.head(10)


# In[21]:


df1 = df1.set_index(df1['DateTime'])


# In[22]:


df1.head(5)


# In[23]:


df1 = df1.drop(['DateTime'],axis=1)


# In[24]:


df1.head(5)


# In[25]:


df2 = df1.copy()


# In[26]:


df2 = df2.dropna()


# In[27]:


df2.isna().sum()


# ##Setting the desired frequency

# In[28]:


df_comp = df2.resample('5Min').mean()


# In[29]:


df_comp.head(90)


# In[30]:


df_comp = df_comp.dropna()


# In[31]:


df2.isna().sum()


# In[32]:


df2.head(3)


# ### Working on df2 as it already resampled to 5 min
# 

# ###If you want to fill missing values, you can use following methods to fill missing values
# df2.Close = df2.Close.fillna(method = 'ffill')
# df2.Close = df2.Close.fillna(method = 'bfill')
# df2.Close = df2.Close.fillna(value = df2.Close.mean())

# ## 

# In[33]:


size = int(len(df2)*0.8)


# In[34]:


df_train = df2.iloc[:size]


# In[35]:


df_test = df2.iloc[size:]


# In[36]:


df_train.tail(5)


# In[37]:


df_test.head(5)


# ## WHITE NOISE
# White noise can't be predicted as it does not follow any pattern
## Three conditions to call a time series as WHITE NOISE:
### 1. Constant Mean = 0
#### 2. Constant Variance
##### 3. No Autocorrelation
# In[38]:


# A time series is white noise if the variables are independent and identically distributed with a mean of zero. 
# This means that all variables have the same variance (sigma^2) and each value has a zero correlation with all other values
# in the series


# In[39]:


#If a time series is white noise, it is a sequence of random numbers and cannot be predicted. If the series of forecast
#errors are not white noise, it suggests improvements could be made to the predictive model.


# In[40]:


wn = np.random.normal(loc = df_train.marketClose.mean(), scale = df_train.marketClose.std(), size = len(df_train))


# In[41]:


df_train['wn'] = wn


# In[42]:


df_train.describe()


# In[43]:


df_train.wn.plot(figsize = (20,5))
plt.title('white noise time series',size=24)
plt.show()


# In[44]:


df_train.head(10)


# In[45]:


df_train.marketClose.plot(figsize = (20,5))
plt.title('SPY ETF Close Prices',size=30)
plt.ylim(260,360)
plt.show()


# # Random Walk Leave now

# ![Random%20Walk.PNG](attachment:Random%20Walk.PNG)

# In[46]:


# rw = pd.read_csv("C:/Users/TAN/Downloads/RandWalk.csv")


# In[47]:


#rw.head(5)


# In[48]:


# rw.date = pd.to_datetime(rw.date,dayfirst = True)
# rw.set_index("date",inplace=True)
# rw = rw.asfreq('b')


# In[49]:


#df_train['rw'] = rw['price']


# In[50]:


#df_train['rw'] = df_train.rw.fillna(value = df_train.rw.mean())


# In[51]:


#df_train = df_train.rw.fillna(df_train.wn.mean())


# In[52]:


#df_train.head(10)


# In[53]:


# df_train.rw.plot(figsize=(20,5))
# plt.title("Random Walk",size=24)
# plt.show()


# # STATIONARITY

# In[54]:


df_train.head(5)


# In[55]:


import statsmodels.tsa.stattools as sts


# In[56]:


sts.adfuller(df_train.marketClose)


# Null hypothesis - Data is not stationary and here  test statistic is greater than critical value so null hypothesis is not rejected.
# Above output shows that data is not stationary
# (P value is 0.59(59% chance that data is not stationary), t statistic > critical values(1%,5%,10%))
# Autocorrelation coefficient is 4 (which is >1)

# In[57]:


sts.adfuller(df_train.wn)


# Above output shows that white noise (wn) data is stationary
# (P value is 0.0(0% chance that data is not stationary-->100% chance that data is stationary)
#  t statistic < critical values(1%,5%,10%))
#  Autocorrelation coefficient is 0 (which is <1)

# In[58]:


#sts.adfuller(rw.price)


# In[59]:


# Same as the closing price


# # Seasonality

# In[60]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[61]:


s_dec_additive = seasonal_decompose(df_train.marketClose, model = "additive",freq=30)


# In[62]:


s_dec_additive.plot()


# In[63]:


####Data is not seasonal as there is not concrete cyclical pattern


# In[64]:


s_dec_multiplicative = seasonal_decompose(df_train.marketClose, model = "multiplicative",freq=30)
s_dec_multiplicative.plot()

# Result -  No seasonality in the SPY ETF Prices
# # Autocorrelation ACF

# In[65]:


import statsmodels.graphics.tsaplots as sgt


# In[66]:


sgt.plot_acf(df_train.marketClose,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title("ACF LAG")


# In[67]:


###Stock prices have been consistent till 40 lags, range of ACF varies from -1 to +1
##Shadow area is the Significance
##We can estimate our prediction even through 40 lags
###The greater the distance in time, the more unlikely that autocorrelation exists
###all the lines are higher than the significance, which means this is the indicator of time dependence in the data
###Prices even a month back can serve as decent estimators
###SPY ETF high frequency data prices is highly autocorrelated


# In[68]:


sgt.plot_acf(df_train.wn,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('WN')
plt.title("WHITE NOISE")


# In[69]:


###no autocorrelation in lags for white noise


# # Partial Autocorrelation function

# In[70]:


sgt.plot_pacf(df_train.marketClose,lags=40,zero=False,alpha=0.5,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF LAG")


# In[71]:


###First lag for ACF and PACF is same because there is no value in between one lag and current price
###After some lags all other values are nearly zero, doesn't have any significance so no affect


# In[72]:


sgt.plot_pacf(df_train.wn,lags=40,zero=False,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF WN")


# In[73]:


###Again proved no autocorrelation in white noise


# # Autoregressive MA model for one lag

# In[74]:


from statsmodels.tsa.arima_model import ARMA


# # AR 1 Model

# In[75]:


model_ar = ARMA(df_train.marketClose,order = (1,0))
# 1 in order represent number of lags, 0 means not taking into residual values into the consideration
# The above is the simple AR(1) model


# In[76]:


results_ar = model_ar.fit()


# In[77]:


results_ar.summary()


# In[78]:


#const is the constant
#ar.L1.market_value is the coefficient of 1 lag ago
#standard error - how far away the model predictions from the true value
#z value - associated test statistics for significance
# p value = 0 means constant and one lag value are both significantly different from zero
# last two columns represent the critical values for 95% confidence interval, if zero is not a part of it we can confirm that
# coefficients are significant


# In[79]:


# Since one lag is significant, we move towards higher lag to build more accurate model which will be more complex


# # Fitting Higher -Lags AR models for Prices

# # AR 2 Model

# In[80]:


model_ar_2 = ARMA(df_train.marketClose,order = [2,0])
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()


# In[81]:


model_ar_3 = ARMA(df_train.marketClose,order = [3,0])
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()


# In[82]:


model_ar_4 = ARMA(df_train.marketClose,order = [4,0])
results_ar_4 = model_ar_4.fit()
results_ar_4.summary()


# In[83]:


model_ar_5 = ARMA(df_train.marketClose,order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())


# In[84]:


# Higher Log Likelihood and lower Information Criteris means better model


# In[85]:


## Log likelihood should increas and information criteria(AIC,BIC,HQIC) should decrease
# p < 0.05 --> lags are significant different and can be used for prediction
# p > 0.05 ---> Insignigicant to predict using this lag


# In[86]:


# Log Likelohood test to compare multiple lag models and decide till what lag we can take


# In[87]:


#In statistics, the likelihood-ratio test assesses the goodness of fit of two competing statistical 
#models based on the ratio of their likelihoods, specifically one found by maximization over 
#the entire parameter space and another found after imposing some constraint
# more lags more better model


# In[88]:


from scipy.stats import chi2
def LLR(mod1,mod2,DF=1):
    L1 = mod1.llf
    L2 = mod2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR,DF).round(3)
    return p


# In[89]:


LLR(results_ar_2,results_ar_3)


# In[90]:


model_ar_5 = ARMA(df_train.marketClose,order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("LLR Test: " + str(LLR(results_ar_2,results_ar_5)))


# In[91]:


model_ar_6 = ARMA(df_train.marketClose,order = (6,0))
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("LLR Test: " + str(LLR(results_ar_2,results_ar_6)))


# In[92]:


model_ar_7 = ARMA(df_train.marketClose,order = (7,0))
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary())
print("LLR Test: " + str(LLR(results_ar_6,results_ar_7)))


# In[93]:


model_ar_8 = ARMA(df_train.marketClose,order = (8,0))
results_ar_8 = model_ar_8.fit()
print(results_ar_8.summary())
print("LLR Test: " + str(LLR(results_ar_7,results_ar_8)))


# In[94]:


model_ar_9 = ARMA(df_train.marketClose,order = (9,0))
results_ar_9 = model_ar_9.fit()
print(results_ar_9.summary())
print("LLR Test: " + str(LLR(results_ar_8,results_ar_9)))


# In[95]:


# Higher lag models are good fit but this is beacuse the prices are highly correlated and not stationary as resulted from 
# Dickey Fuller Test
# We should take returns


# # Returns

# In[96]:


df.head(2)


# In[97]:


df2['returns'] = df2.marketClose.pct_change(1).mul(100)


# In[98]:


df3 = df2.iloc[1:]


# In[99]:


df3.head(10)


# In[100]:


sts.adfuller(df3.returns)


# In[101]:


# Test statistic is far less then the critical values, therefore null hypothesis is rejected --> Returns are stationary


# In[102]:


## Transformed SPY stock close price which were non stationary as tested by adfuller test earlier are transformed
## to stationary series by replacing close price with returns


# In[134]:


## ACF for returns
sgt.plot_acf(df3.returns,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('returns')
plt.title("ACF for returns")


# In[104]:


sgt.plot_pacf(df3.returns,lags=40,zero=False,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF Returns")


# # AR Model For Returns

# In[105]:


model_ret_ar_1 = ARMA(df3.returns,order=(1,0))


# In[106]:


results_ret_ar_1 = model_ret_ar_1.fit()


# In[107]:


results_ret_ar_1.summary()


# In[108]:


# AR2 Model


# In[109]:


model_ret_ar_2 = ARMA(df3.returns,order=(2,0))
results_ret_ar_2 = model_ret_ar_2.fit()
results_ret_ar_2.summary()


# In[110]:


# model_ar_9 = ARMA(df_train.marketClose,order = (9,0))
# results_ar_9 = model_ar_9.fit()
# print(results_ar_9.summary())
print("LLR Test: " + str(LLR(results_ret_ar_1,results_ret_ar_2)))
# for better model one should have high log likelihood, low information criteria, p value less than 0.05


# In[111]:


model_ret_ar_3 = ARMA(df3.returns,order=(3,0))
results_ret_ar_3 = model_ret_ar_3.fit()
print(results_ret_ar_3.summary())
print("LLR Test: " + str(LLR(results_ret_ar_2,results_ret_ar_3)))


# In[112]:


model_ret_ar_4 = ARMA(df3.returns,order=(4,0))
results_ret_ar_4 = model_ret_ar_4.fit()
print(results_ret_ar_4.summary())
print("LLR Test: " + str(LLR(results_ret_ar_3,results_ret_ar_4)))


# In[113]:


model_ret_ar_5 = ARMA(df3.returns,order=(5,0))
results_ret_ar_5 = model_ret_ar_5.fit()
print(results_ret_ar_5.summary())
print("LLR Test: " + str(LLR(results_ret_ar_4,results_ret_ar_5)))


# In[114]:


model_ret_ar_6 = ARMA(df3.returns,order=(6,0))
results_ret_ar_6 = model_ret_ar_6.fit()
print(results_ret_ar_6.summary())
print("LLR Test: " + str(LLR(results_ret_ar_5,results_ret_ar_6)))


# In[115]:


model_ret_ar_7 = ARMA(df3.returns,order=(7,0))
results_ret_ar_7 = model_ret_ar_7.fit()
print(results_ret_ar_7.summary())
print("LLR Test: " + str(LLR(results_ret_ar_6,results_ret_ar_7)))


# In[116]:


model_ret_ar_8 = ARMA(df3.returns,order=(8,0))
results_ret_ar_8 = model_ret_ar_8.fit()
print(results_ret_ar_8.summary())
print("LLR Test: " + str(LLR(results_ret_ar_7,results_ret_ar_8)))


# In[117]:


model_ret_ar_9 = ARMA(df3.returns,order=(9,0))
results_ret_ar_9 = model_ret_ar_9.fit()
print(results_ret_ar_9.summary())
print("LLR Test: " + str(LLR(results_ret_ar_8,results_ret_ar_9)))


# In[118]:


model_ret_ar_10 = ARMA(df3.returns,order=(10,0))
results_ret_ar_10 = model_ret_ar_10.fit()
print(results_ret_ar_10.summary())
print("LLR Test: " + str(LLR(results_ret_ar_9,results_ret_ar_10)))


# # NORMALIZING Actual Price

# In[119]:


#df.head(4)


# In[120]:


# Normalizing market_value first - dividing each value by the benchmark(consider first value) and multiply by 100{This is done
#for easy comparison between time series in percentage}
#benchmark = df.market_value.iloc[0]


# In[121]:


#df['norma'] = df.market_value.div(benchmark).mul(100)


# In[122]:


#sts.adfuller(df.norma)


# In[123]:


## As seen test statistics is higher than critical values and p value is >0.05--> the data is not stationary so we move to 
# calculate the normalized return


# # Normalizing Returns

# In[124]:


#bench_mark = df.returns.iloc[0]


# In[125]:


#df['norm_ret'] = df.returns.div(bench_mark).mul(100)


# In[126]:


#sts.adfuller(df.norm_ret)


# In[127]:


## Test statistics  is less than critical values and p value is <0.05 which shows data is stationary


# In[128]:


## NORMALIZING DOES NOT AFFECT STATIONARITY


# In[ ]:





# # AR MODEL RESIDUALS

# In[150]:


# Analysing the residuals to rpove that residuals follow white noise -
# Do this for both price and returns
# 1. returns should be stationary (check by dickey fuller test)
# 2. we have already calculate the best AR model that fits the data best using LLR test
# 3. check the residuals of the best fitter AR model for both price and returns
# 4. calculate the mean and variance of residuals, they should be around 0 to follow white nose
# 5. Plot ACF to verify that resiudals are not significant because noise don't depend on the previous lagged versions as they are not autocorrelated


# In[129]:


df3.head(10)


# In[130]:


df3['res_price'] = model_ar_9.fit().resid


# # MA MODEL

# In[131]:


sgt.plot_acf(df3.returns[1:],lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('returns')
plt.title("ACF for returns",size=24)


# In[133]:


model_ret_ma_1 = ARMA(df3.returns,order=(0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()


# In[135]:


# order component is 0,1 - (AR component, MA Component)
# 1st order lag is significant
# Now fitting higher models


# In[142]:


model_ret_ma_2 = ARMA(df3.returns,order=(0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
results_ret_ma_2.summary()
print(results_ret_ma_2.summary())
print("LLR Test: " + str(LLR(results_ret_ma_1,results_ret_ma_2)))


# In[138]:


## 2nd lag model is not significant as p value is 0.454
## Checking for higher models


# In[148]:


model_ret_ma_3 = ARMA(df3.returns,order=(0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print("LLR Test: " + str(LLR(results_ret_ma_2,results_ret_ma_3)))


# In[149]:


# Check for higher lagged models - until you get a high significant model
# Regularly do LLR test to compare different models


# In[ ]:




