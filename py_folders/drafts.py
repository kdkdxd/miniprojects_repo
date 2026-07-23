
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f
np.random.seed(1704)

dfapl = pd.DataFrame({
    "Ten":   ["An", "Bình", "Chi", "Duy"],
    "Luong": [15,   20,     18,    12],   # triệu
    "Nam":   [3,    5,      4,     2]
})


def luong_thuc_nhan(rank):
    if rank["Nam"] >=5:
        bonus = rank["Luong"]*1.15
    else :
        bonus = rank["Luong"]*1.05
    return bonus + rank["Luong"]

dfapl["Luong_thuc_nhan"] = dfapl.apply(luong_thuc_nhan, axis = 1)
print(dfapl)


# bro they'll earn a lot of money 
























































print("\n\n\nIt's not hard, It's just new.")
