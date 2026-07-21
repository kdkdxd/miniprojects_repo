
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


print(chi_tieu.argmax())
print(chi_tieu.argmin())

print(f"\n{chi_tieu.idxmax()}")
print(f"\n{chi_tieu.idxmin()}")























print("\n\n\nIt's not hard, It's just new.")
