
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

print("i dont know what to do now :/")
result = anderson(doanhthu_y)
print(f"Anderson-Darling: stat={result.statistic:.4f}")
print(f"Critical values: {result.critical_values}")          # if Anderson-Darling < Critical values < 0.05 => Normal Distribution
print(f"Significance levels: {result.significance_level}")   






print("\n\n\nIt's not hard, It's just new.")
