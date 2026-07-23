
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)

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

result_both = pd.merge(don_hang,khach_hang,on="MaKH",how="outer")
result_inner = pd.merge(don_hang,khach_hang,on="MaKH",how="inner")
result_left = pd.merge(don_hang,khach_hang,on="MaKH",how="left")
result_right = pd.merge(don_hang,khach_hang,on="MaKH",how="right")

print(f"\nResult_both: \n{result_both}")
print(f"\nResult_inner: \n{result_inner}")
print(f"\nResult_left: \n{result_left}")
print(f"\nResult_right: \n{result_right}")
























































print("\n\n\nIt's not hard, It's just new.")
