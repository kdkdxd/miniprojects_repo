
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys

sys.stdout.reconfigure(encoding="utf-8")

score = [ 90, 80, 70 , 65, 48, 400, 600 ] #  

score.append(67)
score.insert(3,77)
score.remove(48)
score.sort(reverse=True)



employees = {
    "E001": {"name": "Nguyen Van A", "dept": "Data", "salary": 15000000},
    "E002": {"name": "Tran Thi B",  "dept": "AI",   "salary": 18000000},
    "E003": {"name": "Le Van C",    "dept": "Data", "salary": 12000000},
    "E004": {"name": "Pham Thi D",  "dept": "AI",   "salary": 20000000},
}


print(employees["E001"]["name"])












































print("\n\n\nIt's not hard, It's just new.")
