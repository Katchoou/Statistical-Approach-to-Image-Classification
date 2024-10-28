# IMporting the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sns
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

##
#Importing the dataset
data=pd.read_excel("D:\\_ECONOMETRICS_\\Datasets\\eco.xlsx", header=0, parse_dates=True, index_col='YEAR')
data.head(1)
#
# Plotting the original data 
plt.figure(figsize=(10, 6))
plt.plot(data.index, data.values, label="FDI Net Inflows into Togo", c='b')
plt.xlabel("Year", fontsize=14, fontname='Times New Roman')
plt.axhline(y=0, color= 'green', linestyle='--')
plt.ylabel('Values', fontsize=14, fontname='Times New Roman')
plt.xticks( fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.grid(True)
plt.show()
###
# Analysis of the ACF and PACF of the data
fdi_acf=plot_acf(data)
fdi_pacf=plot_pacf(data)
plt.figure(figsize=(10, 6))
plt.show()
####
        # Stationarity test of the data
        # ADF Test - Levels: Intercept
adf_interpect=adfuller(data, regression="c")
print("ADF Statistic (Levels: INtercept):", adf_interpect[0])
print("The Pvalue is:", adf_interpect[1])
print("The critical Values are:", adf_interpect[4])
      
        # ADF Test - Levels: Trend & Intercept
adf_trend=adfuller(data, regression="ct")
print("ADF Statistic (Levels: trend & Interpcet):", adf_trend[0])
print("The Pvalue is:", adf_trend[1])
print("The critical Values are:", adf_trend[4])

        # ADF Test - Levels: without interpect $ trend
adf_no_trend=adfuller(data, regression="n")
print("ADF Statistic (Levels: Without intercept and trend):", adf_no_trend[0])
print("The Pvalue is:", adf_no_trend[1])
print("The critical Values are:", adf_no_trend[4])
###
# The ADF test above proved that the series is not stationary so proceed by taking and ploting the first difference
diff_data=data.diff(1).dropna()
plt.figure(figsize=(10, 6))
plt.plot(diff_data, label="PLot of the first difference of the fdi series")
###
        # Stationarity test of the first difference
        # ADF Test - Levels: Intercept
diff_adf_interpect=adfuller(diff_data, regression="c")
print("ADF Statistic (Levels: INtercept):", diff_adf_interpect[0])
print("The Pvalue is:", diff_adf_interpect[1])
print("The critical Values are:", diff_adf_interpect[4])
      
        # ADF Test - Levels: Trend & Intercept
diff_adf_trend=adfuller(diff_data, regression="ct")
print("ADF Statistic (Levels: trend & Interpcet):", diff_adf_trend[0])
print("The Pvalue is:", diff_adf_trend[1])
print("The critical Values are:", diff_adf_trend[4])

        # ADF Test - Levels: without interpect $ trend
diff_adf_no_trend=adfuller(diff_data, regression="n")
print("ADF Statistic (Levels: Without intercept and trend):", diff_adf_no_trend[0])
print("The Pvalue is:", diff_adf_no_trend[1])
print("The critical Values are:", diff_adf_no_trend[4])

# The first diference is stationary at all levels
####
# Analysis of the ACF and PACF of the first difference
diff_fdi_acf=plot_acf(diff_data)
diff_fdi_pacf=plot_pacf(diff_data)
plt.figure(figsize=(10, 6))
plt.show()
###
# Suspected Models ARIMA(1, 1,(1,3)), ARIMA((1,3,4), 1, (1,3,9)), ARIMA(1,1,2)
model=ARIMA(data, order=(3,1,0) )
model_fit=model.fit()
print(model_fit.summary())

###
residuals=model_fit.resid[1:]
residuals.plot(title="Density of the residual of the model", kind='kde')
plt.show()

#
ar_roots=model_fit.arroots
ma_roots=model_fit.maroots
# LB Test : Ho: The model deos not exhibit lack of fit (NO AUTOCORRElTION IN THE ERRORS)
sm.stats.acorr_ljungbox(residuals, return_df=True)

#
model_fit.plot_diagnostics(figsize=(15,10))
plt.show()
#
# ANALYSIS OF THE RESIDUAL SQUARE
residuals_2=plot_pacf(residuals**2, method="ywadjusted", alpha=0.05, lags=25)
residuals_2=plot_pacf(residuals**2, alpha=0.05)
##
# LB Test of the square of the residuals
sm.stats.acorr_ljungbox(residuals**2, return_df=True)

# No serial correlation in the standardized residuals
predictions=model_fit.predict(0)
forecast=model_fit.forecast(13)
forecast.index=forecast.index
values=list(predictions.values)+list(forecast.values)
##
# PLot the predictions of the model over the original series
start_year=1970
end_year=2035
dates=pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='YS')
df=pd.DataFrame({'Years': dates.year})
model_fdi=pd.DataFrame({'values': values}, index=dates)
plt.figure(figsize=(10,6))
plt.plot(model_fdi.index, model_fdi, label='Forecast fdi Net Inflows into Togo', c='r')
plt.plot(data.index, data.values, label='fdi net inflows into Togo', c='b')
plt.xlabel("Year", fontsize=14, fontname='Times New Roman')
plt.ylabel('Values', fontsize=14, fontname='Times New Roman')
plt.xticks( fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.grid(True)
plt.show()
##
# Serial correlation test of the errors: No serial correlation, The Pvalues are given bellow all greater than 0.05
model_fit.test_serial_correlation(method ='ljungbox' )[0][1]
#
# Test for the normality of the residuals
model_fit.test_normality(method=None)
#
# Test for heteroskedasticity: The Pvalue is less than 0.05 so the null hypothesis is rejected, therefore there is heteroskedaticity
model_fit.test_heteroskedasticity(method=None)
#
# Stationarity test of the Errors
        # ADF Test - Levels: Intercept
res_adf_interpect=adfuller(residuals, regression="c")
print("ADF Statistic (Levels: INtercept):", res_adf_interpect[0])
print("The Pvalue is:", res_adf_interpect[1])
print("The critical Values are:", res_adf_interpect[4])
      
        # ADF Test - Levels: Trend & Intercept
res_adf_trend=adfuller(residuals, regression="ct")
print("ADF Statistic (Levels: trend & Interpcet):", res_adf_trend[0])
print("The Pvalue is:", res_adf_trend[1])
print("The critical Values are:", res_adf_trend[4])

        # ADF Test - Levels: without interpect $ trend
res_adf_no_trend=adfuller(residuals, regression="n")
print("ADF Statistic (Levels: Without intercept and trend):", res_adf_no_trend[0])
print("The Pvalue is:", res_adf_no_trend[1])
print("The critical Values are:", res_adf_no_trend[4])