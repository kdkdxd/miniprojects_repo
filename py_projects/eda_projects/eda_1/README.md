
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



----------------------------------------------------------------------------------------------------------------------

# VIETNAMESE

# Tổng quan
Dự án này phân tích hồ sơ nhân viên để trả lời các câu hỏi:
- Lương thay đổi như thế nào theo phòng ban, giới tính và trình độ học vấn?
- Có mối tương quan giữa tuổi và kinh nghiệm không?
- Tỷ lệ thưởng có phụ thuộc vào hiệu suất làm việc không?
- Sự khác biệt lương giữa các nhóm có ý nghĩa thống kê không?

# Dữ liệu
- Nguồn: `employees1.csv`
- Dòng: dữ liệu cấp nhân viên
- Cột chính: `Age`, `Experience`, `Salary`, `Performance`, `Bonus_pct`, `Department`, `Gender`, `Education`

# Yêu cầu và cài đặt

pip install numpy pandas matplotlib seaborn scipy statsmodels
Đặt file employees1.csv cùng thư mục với script.

# Cách chạy
Chạy script chính:

bash
python employees_analysis.py
Script sẽ:

Hiển thị thông tin chung và giá trị thiếu

Điền giá trị thiếu bằng trung vị

Tạo các biểu đồ:

Phân phối của các biến số

Hộp (boxplot) phát hiện ngoại lệ tiền lương

Heatmap tương quan giữa Tuổi và Kinh nghiệm

Biểu đồ tán xạ Lương theo Kinh nghiệm kèm đường xu hướng

Biểu đồ cột lương trung bình theo Phòng ban, Giới tính, Học vấn

Heatmap lương trung bình theo Phòng ban và Học vấn

Biểu đồ tán xạ Tỷ lệ thưởng theo Hiệu suất

Thực hiện kiểm định thống kê:

Kiểm định Shapiro-Wilk để kiểm tra phân phối chuẩn của Lương

Kiểm định Mann-Whitney U so sánh lương nam và nữ

Kiểm định Kruskal-Wallis so sánh lương giữa các phòng ban, sau đó dùng Tukey HSD nếu có ý nghĩa

In kết quả ra màn hình console

# Phát hiện chính
Phân phối lương lệch phải và có ngoại lệ.

Tuổi và kinh nghiệm có tương quan dương mạnh.

Có sự khác biệt lương có ý nghĩa thống kê giữa các phòng ban.

Chênh lệch lương theo giới tính đã được kiểm định; kết quả cho biết sự khác biệt có ý nghĩa hay không.

Trình độ học vấn cao hơn gắn với mức lương trung bình cao hơn.

Tỷ lệ thưởng không thể hiện mối quan hệ tuyến tính rõ ràng với điểm hiệu suất.

# Cấu trúc thư mục
text
employees_analysis/
├── employees1.csv
├── employees_analysis.py
└── README.md

