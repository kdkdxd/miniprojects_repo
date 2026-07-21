
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

chi_tieu = pd.Series({
    "Food":      2500000,
    "Rent":      3000000,
    "Transport": 500000,
    "Entertain": 800000
})

print(chi_tieu)
print(f"Money spent on Food : {chi_tieu["Food"]}")
print(chi_tieu.iloc[3])
print(chi_tieu.iloc[1])

chi_tieu["Expense"] = chi_tieu[chi_tieu>1000000]
print(chi_tieu)
























print("\n\n\nIt's not hard, It's just new.")
