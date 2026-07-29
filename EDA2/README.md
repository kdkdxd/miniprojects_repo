
 <<<<<<< HEAD

# EDA[2]
# E-commerce Orders Analysis
=======
EDA Project [2]
# E-commerce Orders Analysis

>>>>>>> ee29a20329a847b4941221402d1100f15a71c60d
A data analysis project that explores an e-commerce orders dataset to uncover insights about revenue, best-selling products, return rates, and customer behavior.

---

# ENGLISH VER

# Overview
This project analyzes **e-commerce transaction data** using Python (pandas, matplotlib, seaborn).  
The goal is to answer key business questions:
- Which product categories generate the most revenue?
- What are the top 10 best-selling products?
- Which categories and price tiers have higher return rates?
- Is there a relationship between price and returns?

# Dataset
- **Source**: `ecommerce_orders[2].csv`
- **Rows**: Order-level transactional data
- **Key columns**: `OrderDate`, `Category`, `Product`, `Quantity`, `UnitPrice`, `Revenue`, `Returned`, `Rating`, `AgeGroup`

# Requirements & Installation
<<<<<<< HEAD

pip install numpy pandas matplotlib seaborn scipy
=======
bash
pip install numpy pandas matplotlib seaborn scipy

>>>>>>> ee29a20329a847b4941221402d1100f15a71c60d
Place the CSV file in the same directory as the script.

# Usage
Run the main script:

bash
python analysis.py
It will:

Display general data info and missing values

Impute missing values

Produce visualizations:

Revenue & orders by category

Top 10 products by revenue & quantity

Return rate by category and price tier

Print summary statistics to the console

<<<<<<< HEAD

# Key Findings
=======
# Key Findings :
>>>>>>> ee29a20329a847b4941221402d1100f15a71c60d
Best category: The category with highest total revenue is clearly shown in bar charts.

Top product: Identified along with its revenue and order count.

Return rate: Average return rate is ~8%.

Some categories exceed this rate significantly.

Expensive products (>500,000 VND) have notably higher return rates.

# File Structure
text
.
├── ecommerce_orders[2].csv   # Raw dataset
├── analysis.py               # Main analysis script
<<<<<<< HEAD
└── README.md                 # This file
=======
└── README.md                 # This file Sample Visualizations
>>>>>>> ee29a20329a847b4941221402d1100f15a71c60d

# Sample Visualizations
(Charts are displayed when the script is run interactively)

<<<<<<< HEAD
