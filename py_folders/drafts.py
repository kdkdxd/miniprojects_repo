
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import anderson

import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)


#seaborn revision

thang_x = [1,2,3,4,5,6]
doanhthu_y = [150,200,180,250,300,280]

t1,v1 = stats.shapiro(thang_x)
t2,v2 = stats.shapiro(doanhthu_y)
if v1 > 0.05:
    print("Normal")
    if v2 > 0.05:
        print("Normal")
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(endog=doanhthu_y, groups=thang_x, alpha= 0.05)
else:
    print("Not Normal")
print(tukey)

def tier_rate(score):
    if score >=8:
        return "Good"
    elif score >=6:
        return "Medium"
    else:
        print("Low")


print("Im gonna make it baby")



































































