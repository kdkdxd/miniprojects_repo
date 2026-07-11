import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys


sys.stdout.reconfigure(encoding="utf-8")

e_orders_raw = pd.read_csv("ecommerce_orders[2].csv", parse_dates=["OrderDate"])

#G1 GENERAL LOOK

print("==================GENERAL LOOK==================")
print(f"\nSize : {e_orders_raw.shape[0]} rows {e_orders_raw.shape[1]} columns")
print(f"\nDtypes : {e_orders_raw.dtypes}")
print(f"\nFirst 5 rows: {e_orders_raw.head()}")

#G2 DESCRIBE

print(f"\nDescribe: {e_orders_raw.describe().T}")

#G3 MISSING VALUES
miss = e_orders_raw.isna().sum()
miss_pct = (miss/len(e_orders_raw)*100).round(2)
missing_report = pd.DataFrame({
    "Missing Quantity ": miss,
    "Missing Percentage": miss_pct
})
print(f"\nMissing Report : \n{missing_report}")

revenue_median = e_orders_raw["Revenue"].median()
unit_median = e_orders_raw["UnitPrice"].median()


e_orders_raw["AgeGroup"] = e_orders_raw["AgeGroup"].fillna("Unknown")
e_orders_raw["Rating"] = e_orders_raw["Rating"].fillna("Unknown")
e_orders_raw["Revenue"] = e_orders_raw["Revenue"].fillna(revenue_median)
e_orders_raw["UnitPrice"] = e_orders_raw["UnitPrice"].fillna(unit_median)

df = cleaned_ds = e_orders_raw 
print(cleaned_ds)

#DATE TIME
df["Month"] = df["OrderDate"].dt.month
df["MonthName"] = df["OrderDate"].dt.month_name()
df["Quarter"] = df["OrderDate"].dt.quarter

print(df)

cat_stats = df.groupby("Category")["Revenue"].agg(
    total_re= "sum",
    count_re = "count",
    mean_re = "mean",
    median_re = "median"
).round(0).sort_values("total_re",ascending=False)

print(f"\nCategories Stats : \n{cat_stats}")

sum_all_cat = cat_stats["total_re"].sum()
cat_stats["pct_re"] = (cat_stats["total_re"]/sum_all_cat*100).round(0)

print(cat_stats)

if len(cat_stats) > 0:
    fig,ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    sns.barplot(x=cat_stats.index,y=cat_stats["total_re"], palette="Set1", ax = ax, edgecolor = "white")
    for bar, val in zip (ax.patches, cat_stats["total_re"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val/1e6:.2f}", ha = "center", va = "bottom",fontname = "Arial", fontsize = 10, fontweight="bold")
        ax.set_title("Which Category has the biggest Revenue ?", fontname = "Arial", fontsize = 19, fontweight = "bold")
        ax.set_xlabel("Category", fontname = "Arial", fontsize = 14, fontweight = "bold")
        ax.set_ylabel("Total Revenue (Million VND)", fontname = "Arial", fontsize = 14, fontweight = "bold")
    plt.tight_layout()
    plt.show()
 
































































