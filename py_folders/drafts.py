
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import anderson

import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)

# PANDAS REVISION

don_hang = pd.DataFrame({
    "MaDH":     [1,      2,      3,      4],
    "MaKH":     ["KH01", "KH02", "KH01", "KH03"],
    "SanPham":  ["Áo",   "Quần", "Giày", "Mũ"],
    "GiaTri":   [200000, 350000, 500000, 120000]
})

khach_hang = pd.DataFrame({
    "MaKH":    ["KH01", "KH02", "KH03", "KH04"],
    "TenKH":   ["An",   "Bình", "Chi",  "Duy"],
    "ThanhPho":["HCM",  "HN",   "DN",   "HCM"]
})

#CONCAT()
thang1 = pd.DataFrame({"SP": ["A","B"], "Doanh": [100, 200]})
thang2 = pd.DataFrame({"SP": ["A","C"], "Doanh": [150, 180]})
both1and2 = pd.concat([thang1,thang2],ignore_index=True)

print(both1and2)
group_A = both1and2[both1and2["SP"]=="A"]



df14 = pd.DataFrame({
    "SP":     ["Áo", "Quần", "Áo", "Giày", "Quần", "Áo"],
    "Doanh":  [200, 350, 220, 500, 400, 180],
    "Phong":  ["HCM","HN","HCM","DN","HN","HCM"]
})

pivot_table14 = df14.pivot_table(                    # Giống df14.groupby(["Phong", "SP"])["Doanh"].sum()
                                 values = "Doanh",   # Lấy cột "Doanh" để tính toán
                                 index = "Phong",    # Lấy "Phong" làm hàng
                                 columns = "SP",     # Lấy "SP" làm cột
                                 aggfunc = "sum",    # Cách gộp nếu bị trùng
                                 fill_value = 0)     # Nếu ko có data thì thay = 0

print(pivot_table14)














































