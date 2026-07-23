
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f


nv = pd.DataFrame({
    "Ten":    ["An", "Bình", "Chi", "Duy", "Em", "Phúc"],
    "Phong":  ["IT", "KD",   "IT",  "HR",  "KD", "IT"],
    "Luong":  [15, 20, 18, 12, 22, 16],   # triệu
    "Nam":    [3,  5,  4,  2,  6,  1]     # năm kinh nghiệm
})

it = nv[nv["Phong"]=="IT"]
it_high_sa = nv[(nv["Phong"]=="IT")&(nv["Luong"]>15)]

print(f"\nNhan vien IT : \n{it}")
print(f"\nNhan vien IT luong cao : \n{it_high_sa}")

kd_hr = nv[nv["Phong"].isin(["KD","HR"])]

wk = pd.DataFrame({
    "Ten":   ["An", "Bình", "Chi", "Duy", "Em", "Phúc", "Giang"],
    "Phong": ["IT", "KD",   "IT",  "HR",  "KD", "IT",   "HR"],
    "Luong": [15,   20,     18,    12,    22,   16,     14],
    "Nam":   [3,    5,      4,     2,     6,    1,      3]
}).reset_index()
































































print("\n\n\nIt's not hard, It's just new.")
