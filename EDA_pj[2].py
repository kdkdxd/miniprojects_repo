import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys


sys.stdout.reconfigure(encoding="utf-8")
np.random.seed(1704)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
e_orders_raw = pd.read_csv(os.path.join(script_dir, "ecommerce_orders[2].csv"), parse_dates=["OrderDate"])

pd.options.display.float_format = '{:,.0f}'.format   #format :,0f

#G1 GENERAL LOOK
print("==================GENERAL LOOK==================")
print(f"\nSize : {e_orders_raw.shape[0]} rows {e_orders_raw.shape[1]} columns")
print(f"\nDtypes : {e_orders_raw.dtypes}")
print(f"\nFirst 5 rows: \n{e_orders_raw.head()}")

#G2 DESCRIBE & INFO
print(f"\nInfo: {e_orders_raw.info()}")
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


#ANALYSICS REVENUE ACC CATEGORIES
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
top_qty = top10_re["tt_qty"].values[::-1]
top_qty_i = top10_re["tt_qty"].index[::-1]

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


#Return
df["is_returned"] = (df["Returned"]=="Có").astype(int)

rate = (df.groupby("Category").agg(
    returned_count = ("is_returned", "sum"),
    total_ords = ("is_returned","count"),
    return_rate = ("is_returned","mean")
)).sort_values("return_rate",ascending=False)

rate["pct"] = (rate["return_rate"]*100).round(1)

print(rate)


cat = rate.index.tolist()
pcts = rate["pct"].values.tolist()
avg_rate = df["is_returned"].mean()*100
colors_bar = ["#C62828" if p > avg_rate else "#388E3C" for p in pcts]

print(f"\nAvg return rate : {avg_rate}%")   # 8% is company's avg return rate

# Return Diagram acc Category
fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)

cat = rate.index.tolist()
pcts = rate["pct"].values.tolist()
avg_rate = df["is_returned"].mean()*100
colors_bar = ["#fc2323" if p > avg_rate else "#5afc3a" for p in pcts]

axx1= axes[0]
sns.barplot(x=cat,y=pcts, palette=colors_bar, ax=axx1, edgecolor = "white", width = 0.6)
axx1.axhline(avg_rate, ls = "-", lw = 0.7, color = "black", label = f"Company's Avg Return rate ({avg_rate}%)")
for bar, pct in zip(axx1.patches, pcts):
    axx1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+ 0.1, f"{pct}", fontname = "Arial", fontsize = 9, fontweight = "bold", ha = "center", va = "bottom")
axx1.set_title("Return Rate acc Category", fontname  ="Arial", fontsize = 16, fontweight="bold")
axx1.set_xlabel("Category", fontname = "Arial", fontsize = 13, fontweight = "bold")
axx1.set_ylabel("Return Rate (Pct)", fontname = "Arial", fontsize = 13, fontweight = "bold")
axx1.legend(fontsize=9, loc = "upper right")
plt.show


#Return Diagram acc Price
df["price_tier"] = pd.cut(
    df["UnitPrice"],
    bins = [0,200000,500000, float("inf")],
    labels = ["Cheap<200k","Medium","Expensive>500k"]
)

print(df["price_tier"])

tier_rate = (df.groupby("price_tier")["is_returned"].mean().mul(100)).round(1)  #mul : *100 for all price tier
print(f"\nTier rate Returned(pct) : \n{tier_rate}")

colors_tier = ["#5afc3a", "#32ad1a", "#fc2323"]
tier = tier_rate.index.tolist()
tier_pcts = tier_rate.values.tolist()

axx2 = axes[1]
sns.barplot(x=tier,y=tier_pcts,palette=colors_tier, edgecolor = "white", width = 0.6, ax = axx2)
axx2.axhline(y=avg_rate, ls = "--", lw = 0.7, color = "black",label = f"Company's Avg Return rate ({avg_rate}%)" )
for bar2, pct2 in zip(axx2.patches, tier_pcts):
    axx2.text(bar2.get_x()+bar2.get_width()/2, bar2.get_height()+0.1, f"{pct2}", fontname = "Arial", fontsize = 9, fontweight = "bold", ha = "center", va = "bottom")
axx2.set_title("Return Rate acc Price_Tier", fontname = "Arial", fontsize = 16, fontweight = "bold")
axx2.set_xlabel("Price_Tier", fontname = "Arial", fontsize = 13, fontweight = "bold")
axx2.set_ylabel("Return Rate (Pct)", fontname = "Arial", fontsize = 13, fontweight = "bold")
axx2.legend(fontsize = 9, loc = "upper left")
plt.show()

















































print("\n\n\nIt's not hard, It's just new.")





























































