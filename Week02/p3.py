import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess

plt.rcParams['figure.figsize'] = (50, 50)

plt.subplot(6, 1, 1)
ar1 = np.array([1, 0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1);
plot_acf(simulated_data_1, lags=20);
plot_pacf(simulated_data_1, lags=20);

plt.subplot(6, 1, 2)
ar2 = np.array([1, 0.9, 0.6])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2);
plot_acf(simulated_data_2, lags=20);
plot_pacf(simulated_data_2, lags=20);

plt.subplot(6, 1, 3)
ar3 = np.array([1, 0.9, 0.6, 0.3])
ma3 = np.array([1])
AR_object3 = ArmaProcess(ar3, ma3)
simulated_data_3 = AR_object3.generate_sample(nsample=1000)
plt.plot(simulated_data_3);
plot_acf(simulated_data_3, lags=20);
plot_pacf(simulated_data_3, lags=20);

plt.subplot(6, 1, 4)
ar4 = np.array([1])
ma4 = np.array([1, 0.9])
AR_object4 = ArmaProcess(ar4, ma4)
simulated_data_4 = AR_object4.generate_sample(nsample=1000)
plt.plot(simulated_data_4);
plot_acf(simulated_data_4, lags=20);
plot_pacf(simulated_data_4, lags=20);

plt.subplot(6, 1, 5)
ar5 = np.array([1])
ma5 = np.array([1, 0.9, 0.6])
AR_object5 = ArmaProcess(ar5, ma5)
simulated_data_5 = AR_object5.generate_sample(nsample=1000)
plt.plot(simulated_data_5);
plot_acf(simulated_data_5, lags=20);
plot_pacf(simulated_data_5, lags=20);

plt.subplot(6, 1, 6)
ar6 = np.array([1])
ma6 = np.array([1, 0.9, 0.6, 0.4])
AR_object6 = ArmaProcess(ar6, ma6)
simulated_data_6 = AR_object6.generate_sample(nsample=1000)
plt.plot(simulated_data_6);
plot_acf(simulated_data_6, lags=20);
plot_pacf(simulated_data_6, lags=20);
