
# EDA[2]
# E-commerce Orders Analysis
A data analysis project that explores an e-commerce orders dataset to uncover insights about revenue, best-selling products, return rates, and customer behavior.

---

## 📄 English

### 📌 Overview
This project analyzes **e-commerce transaction data** using Python (pandas, matplotlib, seaborn).  
The goal is to answer key business questions:
- Which product categories generate the most revenue?
- What are the top 10 best-selling products?
- Which categories and price tiers have higher return rates?
- Is there a relationship between price and returns?

### 📁 Dataset
- **Source**: `ecommerce_orders[2].csv`
- **Rows**: Order-level transactional data
- **Key columns**: `OrderDate`, `Category`, `Product`, `Quantity`, `UnitPrice`, `Revenue`, `Returned`, `Rating`, `AgeGroup`

### 🛠️ Requirements & Installation
```bash
pip install numpy pandas matplotlib seaborn scipy
Place the CSV file in the same directory as the script.

🚀 Usage
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

📊 Key Findings
Best category: The category with highest total revenue is clearly shown in bar charts.

Top product: Identified along with its revenue and order count.

Return rate: Average return rate is ~8%.

Some categories exceed this rate significantly.

Expensive products (>500,000 VND) have notably higher return rates.

📂 File Structure
text
.
├── ecommerce_orders[2].csv   # Raw dataset
├── analysis.py               # Main analysis script
└── README.md                 # This file
📈 Sample Visualizations
(Charts are displayed when the script is run interactively)

📄 Tiếng Việt
📌 Tổng quan
Dự án phân tích dữ liệu giao dịch thương mại điện tử bằng Python (pandas, matplotlib, seaborn).
Mục tiêu trả lời các câu hỏi kinh doanh chính:

Ngành hàng nào mang lại doanh thu cao nhất?

Top 10 sản phẩm bán chạy nhất là gì?

Ngành hàng nào, phân khúc giá nào có tỷ lệ trả hàng cao?

Có mối liên hệ nào giữa giá sản phẩm và tỷ lệ hoàn trả không?

📁 Dữ liệu
Nguồn: ecommerce_orders[2].csv

Số dòng: Dữ liệu cấp đơn hàng

Cột chính: OrderDate, Category, Product, Quantity, UnitPrice, Revenue, Returned, Rating, AgeGroup

🛠️ Yêu cầu & Cài đặt
bash
pip install numpy pandas matplotlib seaborn scipy
Đặt file CSV cùng thư mục với script.

🚀 Cách chạy
Chạy script chính:

bash
python analysis.py
Script sẽ:

Hiển thị thông tin chung và dữ liệu thiếu

Điền giá trị thiếu

Tạo biểu đồ:

Doanh thu và số đơn theo ngành hàng

Top 10 sản phẩm theo doanh thu và số lượng bán

Tỷ lệ trả hàng theo ngành hàng và phân khúc giá

In số liệu thống kê tổng hợp ra console

📊 Phát hiện chính
Ngành hàng tốt nhất: Ngành có tổng doanh thu cao nhất được làm nổi bật trên biểu đồ.

Sản phẩm đứng đầu: Được xác định kèm doanh thu và số đơn.

Tỷ lệ trả hàng: Trung bình toàn công ty ~8%.

Một số ngành hàng vượt tỷ lệ này đáng kể.

Sản phẩm đắt tiền (trên 500.000 VND) có tỷ lệ trả hàng cao hơn rõ rệt.

📂 Cấu trúc thư mục
text
.
├── ecommerce_orders[2].csv   # Dữ liệu gốc
├── analysis.py               # Script phân tích chính
└── README.md                 # File này
📈 Biểu đồ mẫu
(Các biểu đồ sẽ hiển thị khi chạy script trong môi trường hỗ trợ giao diện đồ họa)