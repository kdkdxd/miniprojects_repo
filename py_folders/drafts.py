
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)

revenueA = np.random.normal(loc=2.0, scale = 0.5, size=50)
revenueB = np.random.normal(loc=2.3, scale = 0.5, size=50)


#t-test 
t1,pvalue1 = stats.ttest_ind(revenueA,revenueB)
print(f"\np_value between Revenue A and Revenue B : {pvalue1:.4f}")
if pvalue1 < 0.05:
    print("Has Statistical Difference")
else:
    print("Doesn't have enough Evidence")



















































print("\n\n\nIt's not hard, It's just new.")
