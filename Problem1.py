import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# General variables
S0 = 120
K = 130
mu = 0
sigma = math.sqrt(255)*0.02
time2mat = 5 / 12
riskfree = 0.08

# Part a
d1 = (math.log(S0 / K) + (mu + sigma ** 2 / 2)
      * time2mat) / sigma * math.sqrt(time2mat)
d2 = d1 - sigma * math.sqrt(time2mat)
P = K * math.exp(-riskfree * time2mat) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
# print(d1)
# print(d2)
print("Put option price is: ", P)

# Part b

deltaT = 10
genprice = np.array([])
norm_dist = np.random.normal(0, 1, 1000000)
genprice = S0*math.exp(1)**((mu-(sigma ** 2)/2) *
                            deltaT+sigma*norm_dist*math.sqrt(deltaT))
profit = genprice - S0

# Drawing the histogram
prft_hist = pd.Series(profit)

bins = np.linspace(-200, 0, 100)
prft_hist.plot.hist(grid=True, bins=bins, range=(-200, 0),
                    density=True, rwidth=0.9, color='#607c8e')
plt.title('PDF of benefit')
plt.ylabel('Probability')
plt.xlabel('Benefit and loss amounts')
plt.grid(axis='y', alpha=0.75)

# Part c
VaR95 = -1*prft_hist.quantile(1-0.95)
print("VaR at 0.95%% is:", VaR95)

n = 1000000
sorted_data = np.sort(prft_hist)
ES = -1*np.sum(sorted_data[0: int(n*0.05)]) / (int(n*0.05))
print("ER at 0.95%% is:", ES)
