
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f


nhan_vien = pd.DataFrame({
    "Ten":       ["An",    "Bình",  "Chi",   "Duy",   "Em"],
    "Phong":     ["IT",    "Kinh doanh", "IT", "HR", "Kinh doanh"],
    "Luong":     [15000000, 20000000, 18000000, 12000000, 22000000],
    "KinhNghiem": [3,       5,        4,        2,        6]
})
print(nhan_vien)

print(f"\nShape: {nhan_vien.shape[0]} rows {nhan_vien.shape[1]} columns")
print(f"\nDtype: {nhan_vien.dtypes}")
print(f"\nInfo: {nhan_vien.info()}")
print(f"\nDescribe: \n{nhan_vien.describe()}")


np.random.seed(42)
df = pd.DataFrame({
    "OrderCode": range(1001),
    "Price": np.random.exponential(500000, 1001).round(),
    "Area":np.random.choice(["HCM", "HN", "DN", "CT"], 1001, p=[0.4, 0.35, 0.15, 0.1])
})


ex_stratified = df.groupby("Area",group_keys=False).sample(
    frac=0.25, random_state= 42
)
print(ex_stratified)










print("\n\n\nIt's not hard, It's just new.")
