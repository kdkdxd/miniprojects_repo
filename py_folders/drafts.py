
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys




diem = np.array([7.5, 8.0, 6.5, 9.0, 7.0, 8.5, 5.5, 9.5])

print(np.var(diem).round(2))
print(np.percentile(diem,25))
print(np.percentile(diem,50))
print(np.percentile(diem,75))


doanh_thu = np.array([
    [120, 150, 200, 180],   # NV A
    [200, 220, 190, 210],   # NV B
    [95,  110, 130, 120],   # NV C
])

# mean each both of each workers
print(np.mean(doanh_thu,axis=0))
print(np.mean(doanh_thu, axis=1))

#  total revenue each workers have made
print(np.sum(doanh_thu,axis=0))

#
































print("\n\n\nIt's not hard, It's just new.")
