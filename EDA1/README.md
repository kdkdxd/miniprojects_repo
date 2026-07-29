
# EDA[1]
# Employees Data Analysis
A data analysis project that explores an employee dataset to uncover insights about salary distributions, correlations, and group differences.

---

# English

# Overview
This project analyzes employee data using Python (pandas, matplotlib, seaborn, scipy). The goal is to answer key business questions:
- What is the salary distribution across departments, genders, and education levels?
- Is there a correlation between age and experience?
- Does bonus percentage depend on performance?
- Are salary differences between groups statistically significant?

# Dataset
- Source: `employees1.csv`
- Rows: Employee-level data
- Key columns: `Age`, `Experience`, `Salary`, `Performance`, `Bonus_pct`, `Department`, `Gender`, `Education`

# Requirements & Installation

pip install numpy pandas matplotlib seaborn scipy statsmodels
Place employees1.csv in the same directory as the script.

# Usage
Run the main script:

bash
python analysis.py
It will:

Display general information and missing values

Impute missing values with medians

Produce visualizations:

Distributions of numerical variables

Boxplot for salary outliers

Heatmap of correlation between Age and Experience

Scatter plot of Salary vs Experience with trend line

Bar charts of mean salary by Department, Gender, and Education

Pivot table heatmap of mean salary by Department and Education

Scatter plot of Bonus percentage vs Performance

Perform statistical tests:

Shapiro-Wilk test for normality of Salary

Mann-Whitney U test comparing male and female salaries

Kruskal-Wallis test comparing salaries across departments, followed by Tukey HSD post-hoc if significant

Print summary statistics and test results to the console

# Key Findings
Salary distribution is right-skewed with outliers.

Age and Experience have a strong positive correlation.

Departments show clear differences in average salary; statistical tests confirm significant differences among at least some departments.

Gender salary difference: test results indicate whether a statistically significant gap exists.

Education level correlates with higher average salary.

Bonus percentage does not show a clear linear relationship with performance score.

# File Structure
text
.
├── employees1.csv          # Raw dataset
├── analysis.py             # Main analysis script
└── README.md               # This file


