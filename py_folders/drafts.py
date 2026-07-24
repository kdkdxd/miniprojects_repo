
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)


#seaborn revision

thang_x = [1,2,3,4,5,6]
doanhthu_y = [150,200,180,250,300,280]

fig,ax = plt.subplots(figsize=(8,5))
sns.lineplot(x=thang_x,y=doanhthu_y, lw=0.7, color = "black", markers="o",mfc="red",ms=7,ax=ax)
ax.set_title("Doanh thu theo Tháng", fontname = "Arial", fontsize = 15, fontweight="bold")
ax.set_xlabel("Tháng", fontname = "Arial", fontsize = 10, fontweight="bold")
ax.set_ylabel("Doanh thu (VND)", fontname = "Arial", fontsize = 10, fontweight="bold")
plt.tight_layout()
plt.show()





















print("\n\n\nIt's not hard, It's just new.")
