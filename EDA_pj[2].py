import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys


sys.stdout.reconfigure(encoding="utf-8")

e_orders_raw = pd.read_csv("ecommerce_orders[2].csv", parse_dates=["OrderDate"])
pd.options.display.float_format = '{:,.0f}'.format   #format :,0f

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
categories = cat_stats.index.tolist()

print(cat_stats)



fig,axes = plt.subplots(1,3,figsize=(15, 5))

#Total Revenue per Categories
ax1 = axes[0] 
sns.barplot(x=cat_stats.index,y=cat_stats["total_re"], palette="Set1", ax = ax1, edgecolor = "white")
for bar1, val1 in zip (ax1.patches, cat_stats["total_re"]):
    ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1.5, f"{val1/1e6:.2f}", ha = "center", va = "bottom",fontname = "Arial", fontsize = 10, fontweight="bold")
ax1.set_title("Which Category has the biggest Revenue ?", fontname = "Arial", fontsize = 15, fontweight = "bold")
ax1.set_xlabel("Category", fontname = "Arial", fontsize = 14, fontweight = "bold")
ax1.set_ylabel("Total Revenue (Million VND)", fontname = "Arial", fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.show

print(f"\nBest seller : {max(categories)} ")
 
#Total orders
ax2 = axes[1]
sns.barplot(x=categories, y = cat_stats["count_re"], palette="Set2", ax=ax2, edgecolor = "white", width = 0.6)
for bar2, val2 in zip(ax2.patches, cat_stats["count_re"]):
    ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1.5, f"{val2}", fontname = "Arial", fontsize = 10, fontweight = "bold", ha = "center", va = "bottom")
ax2.set_title("Total Orders per Category", fontname = "Arial", fontsize = 15, fontweight = "bold")
ax2.set_xlabel("Category", fontname = "Arial", fontsize = 14, fontweight = "bold")
ax2.set_ylabel("Orders", fontname = "Arial", fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.show

#Mean revenue in different categories
ax3 = axes[2]
sns.barplot(x=categories, y = cat_stats["mean_re"], ax = ax3, palette="Set3", edgecolor = "white", width = 0.6)
for bar3,val3 in zip(ax3.patches, cat_stats["mean_re"]):
    ax3.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 1.5, f"{val3/1e6:.2f}", ha = "center", va = "bottom", fontname = "Arial", fontsize = 10, fontweight = "bold")
ax3.set_title("Which cat has the highest mean revenue ?", fontname = "Arial", fontsize = 15, fontweight = "bold")
ax3.set_xlabel("Category", fontname = "Arial", fontsize = 14, fontweight = "bold")
ax3.set_ylabel("Mean Revenue (Million VND)", fontname = "Arial", fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.show

plt.suptitle("Analysis Revenue acc Category", fontname = "Arial", fontsize = 17, fontweight = "bold")
plt.tight_layout()
plt.show


print(df.head(10))

#Top 10 best seller
product_stats1 = df.groupby("Product").agg(
    tt_pd_re = ("Revenue", "sum"),
    tt_qty = ("Quantity","count"),
    tt_ord = ("Revenue", "count")
).round(0).sort_values("tt_pd_re", ascending=False)

print(f"\nTop 10 Products : \n{product_stats1.head(10)}")
top10_re = product_stats1.head(10)

product_stats2 = df.groupby("Product").agg(
    tt_pd_re = ("Revenue", "sum"),
    tt_qty = ("Quantity","count"),
    tt_ord = ("Revenue", "count")
).round(0).sort_values("tt_qty", ascending=False)

print(f"\nTop 10 Best Seller : \n{product_stats2.head(10)}")
top10_quant = product_stats2.head(10)


#Horizontal Barchart 
#Top 10 Revenue

fig, axes = plt.subplots(1,2,figsize=(18,4),constrained_layout=True)

rev_vals = top10_re["tt_pd_re"].values[::-1]
rev_cat = top10_re.index[::-1]

ax1 = axes[0]
ax1.barh(rev_cat, rev_vals, color = "#fcdb03", edgecolor = "white", height = 0.65 )
for bar1,val2 in zip(ax1.patches,rev_vals):
    ax1.text(bar1.get_width() + max(rev_vals)*0.01, bar1.get_y() + bar1.get_height()/2, f"{val2:,.0f}", ha = "left", va = "center", fontname = "Arial", fontsize = 10, fontweight = "bold")
ax1.set_title("Top 10 Revenue", fontname = "Arial", fontsize = 17, fontweight = "bold")
ax1.set_xlabel("Revenue(VND)", fontname = "Arial", fontsize = 15, fontweight = "bold")
ax1.set_ylabel("Category", fontname = "Arial", fontsize = 15, fontweight = "bold")
sns.despine()
plt.show

    
#Top 10 best seller
top_qty = top10_quant["tt_qty"].values[::-1]
top_qty_i = top10_quant["tt_qty"].index[::-1]

ax2 = axes[1]
ax2.barh(top_qty_i,top_qty, color = "#03dbfc", edgecolor = "white", height = 0.65)
for bar2, val2 in zip(ax2.patches, top_qty):
    ax2.text(bar2.get_width() + max(top_qty)*0.01, bar2.get_y() + bar2.get_height()/2, f"{val2:,.0f}", ha = "left", va= "center", fontname = "Arial", fontsize = 10, fontweight = "bold")
ax2.set_title("Top 10 Best Sellers Products", fontname = "Arial", fontsize = 17, fontweight = "bold")
ax2.set_xlabel("Revenue(VND)", fontname = "Arial", fontsize = 15, fontweight = "bold")
ax2.set_ylabel("Category", fontname = "Arial",fontsize = 15, fontweight  = "bold")
sns.despine()
plt.show


#Checking orders Outliers
fig, ax  = plt.subplots(figsize=(8,5),constrained_layout = True)
sns.boxplot(df["Quantity"], color = "#5AF043",ax=ax, flierprops = dict(marker="o",ms=5,mfc="red",mec="white", alpha  = 0.6, label = "Outliers"))
ax.set_title("Orders Outliers", fontname = "Arial", fontsize = 17, fontweight ="bold")
plt.show









































































print("It's not hard, It's just new.")





























































