#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


# In[2]:


import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


df = pd.read_csv("SPY1Min.csv")


# In[4]:


df1 = df.copy()


# In[5]:


df1.head(10)


# In[6]:


df1.describe()


# In[7]:


df1.isna().sum()


# In[8]:


df1.isna().sum()


# In[9]:


df1.marketClose.plot(figsize = (20,5), title = "SPY INTRADAY")
plt.show()


# In[10]:


# Time Series(stock price) data does not follow normal distribution because it does not follow Gauss-Markov assumption


# In[11]:


import scipy.stats
import pylab


# ### X axis - Theoritical quantiles is how many standard deviaiton away from mean
# ### Y axis - Stock Prices
# 

# In[12]:


# Quantile-Quantile plot(QQ plot) is used to check whether the distribution is normal or not


# In[13]:


scipy.stats.probplot(df1.marketClose,plot = pylab)
pylab.show()


# In[14]:


### Normally distributed as it is following linear graph


# In[15]:


#df1["DateTime"] =  df1['Date'] + ' ' + df1['Time']


# In[16]:


df1.head(10)


# In[17]:


df1['DateTime'] = pd.to_datetime(df1.DateTime,dayfirst = True)


# In[18]:


df1.head(10)


# In[19]:


df1.DateTime.describe()


# In[20]:


#df1 = df1.drop(['Date','Time'],axis=1)


# In[21]:


#df1.head(10)


# In[22]:


df1 = df1.set_index(df1['DateTime'])


# In[23]:


df1.head(5)


# In[24]:


df1 = df1.drop(['DateTime'],axis=1)


# In[25]:


df1.head(5)


# In[26]:


df2 = df1.copy()


# In[27]:


df2 = df2.dropna()


# In[28]:


df2.isna().sum()


# ##Setting the desired frequency

# In[29]:


df_comp = df2.resample('5Min').mean()


# In[30]:


df_comp.head(90)


# In[31]:


df_comp = df_comp.dropna()


# In[32]:


df2.isna().sum()


# In[33]:


df2.head(3)


# ### Working on df2 as it already resampled to 5 min
# 

# If you want to fill missing values, you can use following methods to fill missing values
# df2.Close = df2.Close.fillna(method = 'ffill')
# df2.Close = df2.Close.fillna(method = 'bfill')
# df2.Close = df2.Close.fillna(value = df2.Close.mean())

# In[34]:


size = int(len(df2)*0.8)


# In[35]:


df_train = df2.iloc[:size]


# In[36]:


df_test = df2.iloc[size:]


# In[37]:


df_train.tail(5)


# In[38]:


df_test.head(5)


# ## WHITE NOISE
# White noise can't be predicted as it does not follow any pattern
## Three conditions to call a time series as WHITE NOISE:
### 1. Constant Mean = 0
#### 2. Constant Variance
##### 3. No Autocorrelation
# In[39]:


# A time series is white noise if the variables are independent and identically distributed with a mean of zero. 
# This means that all variables have the same variance (sigma^2) and each value has a zero correlation with all other values
# in the series


# In[40]:


#If a time series is white noise, it is a sequence of random numbers and cannot be predicted. If the series of forecast
#errors are not white noise, it suggests improvements could be made to the predictive model.


# In[41]:


wn = np.random.normal(loc = df_train.marketClose.mean(), scale = df_train.marketClose.std(), size = len(df_train))


# In[42]:


df_train['wn'] = wn


# In[43]:


df_train.describe()


# In[44]:


df_train.wn.plot(figsize = (20,5))
plt.title('white noise time series',size=24)
plt.show()


# In[45]:


df_train.head(10)


# In[46]:


df_train.marketClose.plot(figsize = (20,5))
plt.title('SPY ETF Close Prices',size=30)
plt.ylim(260,360)
plt.show()


# # Random Walk Leave now

# ![Random%20Walk.PNG](attachment:Random%20Walk.PNG)

# # STATIONARITY

# In[47]:


df_train.head(5)


# In[48]:


import statsmodels.tsa.stattools as sts


# In[49]:


sts.adfuller(df_train.marketClose)


# Null hypothesis - Data is not stationary and here  test statistic is greater than critical value so null hypothesis is not rejected.
# Above output shows that data is not stationary
# (P value is 0.59(59% chance that data is not stationary), t statistic > critical values(1%,5%,10%))
# Autocorrelation coefficient is 4 (which is >1)

# In[50]:


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

# In[52]:


s_dec_additive = seasonal_decompose(df_train.marketClose, model = "additive",freq=30)


# In[44]:


s_dec_additive.plot()


# In[45]:


####Data is not seasonal as there is not concrete cyclical pattern


# In[46]:


s_dec_multiplicative = seasonal_decompose(df_train.marketClose, model = "multiplicative",freq=30)
s_dec_multiplicative.plot()

# Result -  No seasonality in the SPY ETF Prices
# # Autocorrelation ACF

# In[69]:


# A plot of the autocorrelation of a time series by lag is called the AutoCorrelation Function, or the acronym ACF. 
# This plot is sometimes called a correlogram or an autocorrelation plot. ... Running the example creates a 2D plot showing the
# lag value along the x-axis and the correlation on the y-axis between -1 and 1


# In[54]:


sgt.plot_acf(df_train.marketClose,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title("ACF LAG")


# In[72]:


###Stock prices have been consistent till 40 lags, range of ACF varies from -1 to +1
##Shadow area is the Significance
##We can estimate our prediction even through 40 lags
###The greater the distance in time, the more unlikely that autocorrelation exists
###all the lines are higher than the significance, which means this is the indicator of time dependence in the data
###Prices even a month back can serve as decent estimators
###SPY ETF high frequency data prices is highly autocorrelated


# In[73]:


sgt.plot_acf(df_train.wn,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('WN')
plt.title("WHITE NOISE")


# In[94]:


###no autocorrelation in lags for white noise
# White noise represents residuals which should be highly significant as shown by significance level
# It means they are not similar or dependent on the past values


# # Partial Autocorrelation function

# In[75]:


sgt.plot_pacf(df_train.marketClose,lags=40,zero=False,alpha=0.5,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF LAG")


# In[76]:


###First lag for ACF and PACF is same because there is no value in between one lag and current price
###After some lags all other values are nearly zero, doesn't have any significance so no affect


# In[77]:


sgt.plot_pacf(df_train.wn,lags=40,zero=False,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF WN")


# In[78]:


###Again proved no autocorrelation in white noise


# # Autoregressive MA model for one lag

# In[79]:


# #An autoregressive model is when a value from a time series is regressed on previous values from that same time series. ... 
# The order of an autoregression is the number of immediately preceding values in the series that are used to predict the value
# at the present time.


# # AR 1 Model

# In[81]:


model_ar = ARMA(df_train.marketClose,order = (1,0))
# 1 in order represent number of lags, 0 means not taking into residual values into the consideration
# The above is the simple AR(1) model


# In[82]:


results_ar = model_ar.fit()


# In[83]:


results_ar.summary()


# In[ ]:


# Always remember - model with lower AIC, BIC and higher log likelihood is better 


# In[84]:


#const is the constant
#ar.L1.market_value is the coefficient of 1 lag ago
#standard error - how far away the model predictions from the true value
#z value - associated test statistics for significance
# p value = 0 means constant and one lag value are both significantly different from zero
# last two columns represent the critical values for 95% confidence interval, if zero is not a part of it we can confirm that
# coefficients are significant


# In[85]:


# Since one lag is significant, we move towards higher lag to build more accurate model which will be more complex


# # Fitting Higher -Lags AR models for Prices

# # AR 2 Model

# In[86]:


model_ar_2 = ARMA(df_train.marketClose,order = [2,0])
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()


# In[87]:


model_ar_3 = ARMA(df_train.marketClose,order = [3,0])
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()


# In[88]:


model_ar_4 = ARMA(df_train.marketClose,order = [4,0])
results_ar_4 = model_ar_4.fit()
results_ar_4.summary()


# In[89]:


model_ar_5 = ARMA(df_train.marketClose,order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())


# In[90]:


# Higher Log Likelihood and lower Information Criteris means better model


# In[91]:


## Log likelihood should increas and information criteria(AIC,BIC,HQIC) should decrease
# p < 0.05 --> lags are significant different and can be used for prediction
# p > 0.05 ---> Insignigicant to predict using this lag


# In[92]:


# Log Likelohood test to compare multiple lag models and decide till what lag we can take


# In[93]:


#In statistics, the likelihood-ratio test assesses the goodness of fit of two competing statistical 
#models based on the ratio of their likelihoods, specifically one found by maximization over 
#the entire parameter space and another found after imposing some constraint
# more lags more better model


# In[34]:


from scipy.stats import chi2
def LLR(mod1,mod2,DF=1):
    L1 = mod1.llf
    L2 = mod2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR,DF).round(3)
    return p


# In[95]:


LLR(results_ar_2,results_ar_3)


# In[96]:


model_ar_5 = ARMA(df_train.marketClose,order = (5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("LLR Test: " + str(LLR(results_ar_2,results_ar_5)))


# In[97]:


model_ar_6 = ARMA(df_train.marketClose,order = (6,0))
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("LLR Test: " + str(LLR(results_ar_2,results_ar_6)))


# In[98]:


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


# In[99]:


# Higher lag models are good fit but this is beacuse the prices are highly correlated hence not stationary as resulted from 
# Dickey Fuller Test
# We should take returns


# # Returns

# In[34]:


df.head(2)


# In[57]:


df2['returns'] = df2.marketClose.pct_change(1).mul(100)


# In[58]:


df3 = df2.iloc[1:]


# In[59]:


df3.head(10)


# In[60]:


sts.adfuller(df3.returns)


# In[47]:


# Test statistic is far less then the critical values, therefore null hypothesis is rejected --> Returns are stationary


# In[110]:


## Transformed SPY stock close price which were non stationary as tested by adfuller test earlier are transformed
## to stationary series by replacing close price with returns


# In[111]:


## ACF for returns
sgt.plot_acf(df3.returns,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('returns')
plt.title("ACF for returns")


# In[112]:


sgt.plot_pacf(df3.returns,lags=40,zero=False,method = 'ols')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.title("PACF Returns")


# # AR Model For Returns

# In[113]:


model_ret_ar_1 = ARMA(df3.returns,order=(1,0))


# In[114]:


results_ret_ar_1 = model_ret_ar_1.fit()


# In[115]:


results_ret_ar_1.summary()


# In[116]:


# AR2 Model


# In[117]:


model_ret_ar_2 = ARMA(df3.returns,order=(2,0))
results_ret_ar_2 = model_ret_ar_2.fit()
results_ret_ar_2.summary()


# In[118]:


# model_ar_9 = ARMA(df_train.marketClose,order = (9,0))
# results_ar_9 = model_ar_9.fit()
# print(results_ar_9.summary())
print("LLR Test: " + str(LLR(results_ret_ar_1,results_ret_ar_2)))
# for better model one should have high log likelihood, low information criteria, p value less than 0.05


# In[119]:


model_ret_ar_3 = ARMA(df3.returns,order=(3,0))
results_ret_ar_3 = model_ret_ar_3.fit()
print(results_ret_ar_3.summary())
print("LLR Test: " + str(LLR(results_ret_ar_2,results_ret_ar_3)))


# In[120]:


model_ret_ar_4 = ARMA(df3.returns,order=(4,0))
results_ret_ar_4 = model_ret_ar_4.fit()
print(results_ret_ar_4.summary())
print("LLR Test: " + str(LLR(results_ret_ar_3,results_ret_ar_4)))


# In[121]:


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


# In[61]:


model_ret_ar_8 = ARMA(df3.returns,order=(8,0))
results_ret_ar_8 = model_ret_ar_8.fit()
print(results_ret_ar_8.summary())
#print("LLR Test: " + str(LLR(results_ret_ar_7,results_ret_ar_8)))


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


# # AR MODEL RESIDUALS

# In[150]:


# Analysing the residuals to rpove that residuals follow white noise -
# Do this for both price and returns
# 1. returns should be stationary (check by dickey fuller test)
# 2. we have already calculate the best AR model that fits the data best using LLR test
# 3. check the residuals of the best fitter AR model for both price and returns
# 4. calculate the mean and variance of residuals, they should be around 0 to follow white nose
# 5. Plot ACF to verify that resiudals are highly significant(means they are not same) because noise don't depend on the previous lagged versions as they are not autocorrelated


# In[129]:


df3.head(10)


# In[64]:


df3['res_price'] = results_ret_ar_8.resid


# # MA MODEL

# In[65]:


df3.head(3)


# In[66]:


sgt.plot_acf(df3.returns[1:],lags=10,zero=False)
plt.xlabel('Lags')
plt.ylabel('returns')
plt.title("ACF for returns",size=24)


# In[133]:


# Returns are stationary means they are not autocorrelated as shown by above ACF


# In[124]:


model_ret_ma_1 = ARMA(df3.returns,order=(0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()


# In[135]:


# order component is 0,1 - (AR component, MA Component)
# 1st order lag is significant
# Now fitting higher models


# In[134]:


model_ret_ma_2 = ARMA(df3.returns,order=(0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
results_ret_ma_2.summary()
print(results_ret_ma_2.summary())
#print("LLR Test: " + str(LLR(results_ret_ma_1,results_ret_ma_2)))


# In[138]:


## 2nd lag model is not significant as p value is 0.454
## Checking for higher models


# In[149]:


# Check for higher lagged models - until you get a high significant model
# Regularly do LLR test to compare different models


# In[137]:


# Skipping some videos on MA Models


# # ARMA

# In[138]:


# The ARMA models contain the both past values(like the AR) and past errors(like the MA)


# In[71]:


model_ret_ar_1_ma_1 = ARMA(df3.returns,order=(1,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
results_ret_ar_1_ma_1.summary()
#print(results_ret_ma_2.summary())


# In[141]:


print("LLR Test MA VS ARMA: " + str(LLR(results_ret_ma_1,results_ret_ar_1_ma_1)))
print("LLR Test AR VS ARMA: " + str(LLR(results_ret_ar_1,results_ret_ar_1_ma_1)))


# In[ ]:


# The above LLR test proves that MA1 model was better than ARMA 1 (MA model was useful in predict present residual based on
# past residual)
# However ARMA1 was better than MA1(MA works on values not residual)


# In[74]:


model_ret_ar_2_ma_2 = ARMA(df3.returns,order=(2,2))
results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()
#results_ret_ar_2_ma_2.summary()


# In[142]:


print("LLR Test MA VS ARMA: " + str(LLR(results_ret_ma_2,results_ret_ar_2_ma_2)))
print("LLR Test AR VS ARMA: " + str(LLR(results_ret_ar_2,results_ret_ar_2_ma_2)))


# In[ ]:


# ARMA2 is btter value predictor than AR 2
# MA2 is better residual predictor than ARMA2


# In[143]:


model_ret_ar_8_ma_6 = ARMA(df3.returns,order=(8,6))
results_ret_ar_8_ma_6 = model_ret_ar_8_ma_6.fit()
results_ret_ar_8_ma_6.summary()


# In[62]:


# Observing MA coeffecients which are not significant with model ma.L5 and above as the p values are above 0.05, we can move to
# simpler lag ma model till MA4
# With AR model, the coefficients are not telling a very straightforward story as some models are signficant while others are not
# Lets take a model half such as ARMA(3,3)


# In[72]:


model_ret_ar_3_ma_3 = ARMA(df3.returns,order=(3,3))
results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()


# In[73]:


LLR(results_ret_ar_1_ma_1,results_ret_ar_3_ma_3,DF=4)


# In[75]:


results_ret_ar_3_ma_3.summary()


# In[76]:


LLR(results_ret_ar_2_ma_2,results_ret_ar_3_ma_3,DF=2)


# In[77]:


model_ret_ar_1_ma_2 = ARMA(df3.returns,order=(1,2))
results_ret_ar_1_ma_2 = model_ret_ar_1_ma_2.fit()


# In[78]:


LLR(results_ret_ar_1_ma_1,results_ret_ar_1_ma_2,DF=1)


# In[80]:


LLR(results_ret_ar_2_ma_2,results_ret_ar_1_ma_2,DF=1)


# In[79]:


results_ret_ar_1_ma_2.summary()


# In[82]:


# ARMA(1,2) model has resulted in the most efficient model
# Also absolute values of coefficient for ma model are decreased from 1.8823 to 0.8852, this means that residuals are reduced


# # Residuals for ARMA

# In[95]:


df3['resid_results_ret_ar_1_ma_2'] = results_ret_ar_1_ma_2.resid


# In[97]:


df3.resid_results_ret_ar_1_ma_2.plot(figsize=(20,5))
plt.title('Residuals of returns',size=24)
plt.show()


# In[98]:


sgt.plot_acf(df3.resid_results_ret_ar_1_ma_2,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF of residuals of returns for ARMA(1,2)')
plt.title("ACF LAG")


# # ACF is helpful for understanding if residuals represesnt White Noise

# In[99]:


# As all the residuals are within significance level we can say that all are significant or different or not autocorrelated or
# represents white noise


# In[100]:


# ARMA Model works better while modeling stationary time series data such as returns
# as the log likelihood for returns is higher than that of market prices


# # ARIMA(Autoregressive Integrated Moving Average)

# ### To model non stationary data, ARIMA(Autoregressive Integrated Moving Average) model comes into picture

# #### Models the difference between prices(across periods) instead of pricee. The purpose of the model is to ensure stationarity no matter what the underlying data is.

# #### ARIMA model is nothing but an ARMA model applied on the newly generated time series from the difference in prices

# ### ARIMA (1,1,1) - Given 2nd parameter (Integration Order) as 1 that it will automatically take the 1 period lag difference in dataset.

# In[41]:


df3.head()


# In[42]:


model_ar_1_i_1_ma_1 = ARIMA(df3.marketClose,order=(1,1,1))
result_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
result_ar_1_i_1_ma_1.summary()


# In[ ]:


# MA1 is significant but AR1 is not significant , may be higher lag models are better
# If you see returns for ARMA (1,1) it is similar to what we see now in ARIMA(1,1,1) , it is because ARIMA is tranforming the 
# prices into stationary form by taking differences instead of calculating returns


# ### ARIMA (1,1,1) Residuals

# In[43]:


df3['res_ar_1_i_1_ma_1'] = result_ar_1_i_1_ma_1.resid
sgt.plot_acf(df3['res_ar_1_i_1_ma_1'],zero=False, lags=40)
plt.title("ACF for residuals for ARIMA(1,1,1)",size=20)


# In[45]:


# The ACF failed to compute due to the missing first element
# Remove the first row of the dataframe


# In[46]:


#df3 = df3.iloc[1:]


# In[50]:


df3['res_ar_1_i_1_ma_1'] = result_ar_1_i_1_ma_1.resid
sgt.plot_acf(df3['res_ar_1_i_1_ma_1'][1:],zero=False, lags=40)
plt.title("ACF for residuals for ARIMA(1,1,1)",size=20)


# In[51]:


# Including 4th lag in our model will improve our performance


# In[52]:


# Higher lag models


# In[ ]:


# ARIMA(3,4)


# In[53]:


model_ar_3_i_1_ma_4 = ARIMA(df3.marketClose,order=(3,1,4))
result_ar_3_i_1_ma_4 = model_ar_3_i_1_ma_4.fit()
result_ar_3_i_1_ma_4.summary()


# In[56]:


df3['res_ar_3_i_1_ma_4'] = result_ar_3_i_1_ma_4.resid
sgt.plot_acf(df3['res_ar_3_i_1_ma_4'][1:],zero=False, lags=40)
plt.title("ACF for residuals for ARIMA(3,1,4)",size=20)


# In[61]:


# Higher model ARIMA(3,1,4) is highly significant so we check for LLR test to compare two models
LLR(result_ar_1_i_1_ma_1,result_ar_3_i_1_ma_4,DF=2)


# In[62]:


# at 5% level, ARIMA (3,1,4) has high significance than ARIMA model (1,1,1)


# ### ARIMA (4,1,4)

# In[63]:


# Checking for higher lag model
# G
model_ar_4_i_1_ma_4 = ARIMA(df3.marketClose,order=(4,1,4))
result_ar_4_i_1_ma_4 = model_ar_4_i_1_ma_4.fit()
result_ar_4_i_1_ma_4.summary()


# In[64]:


df3['res_ar_4_i_1_ma_4'] = result_ar_4_i_1_ma_4.resid
sgt.plot_acf(df3['res_ar_4_i_1_ma_4'][1:],zero=False, lags=40)
plt.title("ACF for residuals for ARIMA(4,1,4)",size=20)


# In[65]:


LLR(result_ar_3_i_1_ma_4,result_ar_4_i_1_ma_4,DF=1)


# In[66]:


# Reasons why ARIMA(4,1,4) is better than ARIMA(3,1,4)
# 1. Higher Log likelihood
# 2. Lower AIC, BIC
# 3. ACF - all lags in sigificance level
# 4. LLR - ARIMA(4,1,4) vs ARIMA (3,1,4) - 0.0


# ### Higher Integration ARIMA Model

# In[50]:


## For higher integration, we need difference between index prices from period to period

### First to check if we need higher integration model , we calculate if integrated data has been stationary in ARIMA(x,1,z)model


# In[56]:


df3.head()


# In[57]:


df3['delta_prices'] = df3.marketClose.diff(1)


# In[60]:


### check if generated integrated data is correct by comparing ARIMA(1,0,1) on delta prices and ARMA(1,1) on normal prices, they 
# both should be equal


# In[59]:


model_delta_ar_1_i_1_ma_1 = ARIMA(df3.delta_prices[1:],order=(1,0,1))
result_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
result_delta_ar_1_i_1_ma_1.summary()C:\Users\TAN\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:219: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.


# In[61]:


## Checking stationary of the integrated data
sts.adfuller(df3.delta_prices[1:])


# In[62]:


## Null hypothesis - Data is not stationary
## P value --> 0.0 and test stattistic(-53.39) is less than the critical values , null hypothesis is rejected
## Integrated Prices are stationary


# In[ ]:


## ARIMA is more computationally expensive as it has to calculate the integrated diffference in period 


# # ARCH Model (Autoregressive Conditional Heteroskedasticity)

# In[ ]:


# ARCH models attempt to model the variance of these error terms, and in the process correct for the problems resulting 
# from heteroskedasticity. The goal of ARCH models is to provide a measure of volatility that can be used in financial 
# decision-making.


# In[ ]:


### Heteroskedasticity - Not uniform dispersion or Variance is not uniform


# In[63]:


df3.head()


# In[65]:


## Squared Returns
df3['sqd_returns'] = df3.returns.mul(df3.returns)


# In[66]:


df3.head()


# In[67]:


df3.returns.plot(figsize=(20,5))
plt.title("Returns",size=24)


# In[68]:


df3.sqd_returns.plot(figsize=(20,5))
plt.title("Squared Returns",size=24)


# In[72]:


sgt.plot_acf(df3['returns'][1:],zero=False, lags=40)
plt.title("ACF Returns",size=20)


# In[76]:


sgt.plot_acf(df3['sqd_returns'][1:],lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title("ACF Squared Returns")


# In[77]:


from arch import arch_model


# In[79]:


model_arch_1 = arch_model(df3['returns'][1:])
results_arch_1 = model_arch_1.fit()
results_arch_1.summary()


# In[80]:


# Mean Model = Constant --> Mean is contant rather than moving which is the property of the stationary data
# Vol Model = GARCH ---> It is using GARCH model to model the variance
# Dd Model = four variables are calculated
# Mean Model:
#     coeff of mean in the equation
#     higher t value and p <0.05 determines significance of coefficient
# Volatiliy Model:
#     omega is alpha 0
#     alpha 1
#     beta:
# In Iteration table - total no of iteration to converge
    


# In[81]:


model_arch_1 = arch_model(df3['returns'][1:],mean="Constant",vol='ARCH',p=1)
results_arch_1 = model_arch_1.fit(update_freq=5)
results_arch_1.summary()


# In[89]:


## When we give mean as 'AR' it means that the mean is dependent on past values as in the case of AR model that also means that
## mean can be autocorrelated, and actually below model is better than using constant mean model
model_arch_1_AR = arch_model(df3['returns'][1:],mean="AR",lags=[2,3,6],vol='ARCH',p=1)
results_arch_1_AR = model_arch_1_AR.fit(update_freq=5)
results_arch_1_AR.summary()


# In[82]:


## Significant Model is obtained check for higher lag model


# In[83]:


model_arch_2 = arch_model(df3['returns'][1:],mean="Constant",vol='ARCH',p=2)
results_arch_2 = model_arch_2.fit(update_freq=5)
results_arch_2.summary()


# In[90]:


model_arch_2_AR = arch_model(df3['returns'][1:],mean="AR",lags=[2,3,6],vol='ARCH',p=2)
results_arch_2_AR = model_arch_2_AR.fit(update_freq=5)
results_arch_2_AR.summary()


# In[87]:


### Higher lag (2nd order) model is not significant as Log Likelihood has converged, also AIC and BIC both have increased
### instead of decreasing and p value of alpha 2 coefficiant is not significant 


# # GARCH Model

# In[88]:


## Including previous values in ARCH model improve the model
## These previous values can't be returns because mean model is already taking that into consideration
## Due to volatility Clustering - high volatility is followed by high volatility and low volatilty is followed by low volatility
## If we include previous variance to model current variance it will improve ARCH model
## This is the reason we use Garch Model


# ![GARCH.png](attachment:GARCH.png)

# ## GARCH (1,1) with Serially Uncorrelated mean or using constant mean model

# In[93]:


# p is Squared residual for the past period as show above
# q is conditonal variance from last period
model_garch_1_1 = arch_model(df3.returns[1:],mean='Constant',vol='GARCH',p=1,q=1)
results_garch_1_1 = model_garch_1_1.fit(update_freq=5)
results_garch_1_1.summary()


# # Auto ARIMA Model - Automates the ARIMA model selection

# In[ ]:


# First it selects the parameter AIC or BIC
# Then It fits different model and check which model has lowest AIC, BIC (AIC or BIC include log likelihood in the forumula)


# In[39]:


df3.head()


# In[40]:


from pmdarima.arima import auto_arima


# In[41]:


model_auto = auto_arima(df3.returns[1:])


# In[42]:


model_auto


# In[43]:


model_auto.summary()


# In[44]:


## Default best model for returns is MA1 model


# In[ ]:




