
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


# paired-test

doanh_so_truoc = np.array([15, 18, 12, 20, 16, 14, 22, 19])
doanh_so_sau   = np.array([18, 20, 15, 22, 18, 17, 24, 21])

t2,pvalue2 = stats.ttest_rel(doanh_so_truoc,doanh_so_sau)
if pvalue2 < 0.05:
    print("Training is Efficient")
else:
    print("Training is Inefficient")

# pearsonr (tương quan)
ex_pear = np.random.randint(1,15,50)
salary_pear = ex_pear*1.5 + np.random.randn(50)*2 + 10

r,pval_pear = stats.pearsonr(ex_pear,salary_pear)
print(f"\nr: {r}")
print(f"\nPearsonr p-value {pval_pear:.4f}")
if pval_pear < 0.05:
    if r >= 0.07:
        print("Tương quan mạnh")
    elif r >= 0.03:
        print("Tương quan vừa")
    else:
        print("Tương quan yếu")
else:
    print("Chưa đủ bằng chứng thống kê")


# Shapiro-Wilk test (tốt cho n < 2000)
# H₀: dữ liệu có phân phối chuẩn


data_chuan = np.random.normal(0,1,100)
data_lech = np.random.exponential(1,100)

stat1, p1 = stats.shapiro(data_chuan)
stat2, p2 = stats.shapiro(data_lech)

print(f"Data chuẩn  → p={p1:.4f}", "✅ Chuẩn" if p1 > 0.05 else "❌ Không chuẩn")
print(f"Data lệch   → p={p2:.4f}", "✅ Chuẩn" if p2 > 0.05 else "❌ Không chuẩn")

#list

list = [1,2,3,4,5,6,7,8,9,10]
print(list[1])
list = list.append([19,63,84,97])
print("I'm smart.")






































print("\n\n\nIt's not hard, It's just new.")
