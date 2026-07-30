import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Nạp dataset THẬT
data = load_diabetes()
X = data.data
y = data.target
feature_names = list(data.feature_names)
print(f"Shape X (features) : {X.shape}")
print(f"Shape y (target) : {y.shape}")
print(f"\n10 feature names : {feature_names}")
print(f"First sample (patient) : {X[0]}")
print(f"Real Target of first patient : {y[0]}")

# PHẦN 1 : EDA (EXPLORATORY DATA ANALYSIS)
df = pd.DataFrame(X, columns= feature_names)
df['Target'] = y
print(f"\nFirst 5 rows : \n{df.head()}")
print(f"\nThống kê mô tả (Describe, 10 dòng Feature + Target) : \n{df.describe().round(4)}")


# Tính tương quan (correlation) của từng feature với target - đây là bước
# quan trọng để biết feature nào "liên quan" nhiều nhất đến điều ta muốn dự
# đoán, TRƯỚC KHI huấn luyện bất kỳ mô hình nào.

corr_with_target = df.corr()['Target'].sort_values(ascending=False)
print(f"\nHệ số tương quan giữa từng feature và Target:")
print(corr_with_target.round(4))

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,fmt='.2f', cmap="coolwarm",center=0)
plt.title("Ma tran tuong quan giua 10 features va target", fontname="Arial",fontsize=15,fontweight="bold")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

print(f"X_train size : {X_train.shape}")
print(f"X_test size : {X_test.shape}")
print(f"y_train size : {y_train.shape}")
print(f"y_test size : {y_test.shape}")

# Cost Function
def compute_cost(x, y, w, b):
    """
    Tinh gia tri ham mat mat MSE cho Simple Linear Regression (1 feature).

    Tham so:
        x : mang numpy 1 chieu, cac gia tri feature (o day la bmi)
        y : mang numpy 1 chieu, cac gia tri target THAT
        w, b : 2 tham so (so thuc) cua duong thang y_hat = w*x + b

    Tra ve:
        cost : 1 so thuc duy nhat, do "do te" cua duong thang (w, b) do
    """
    m = len(x)                    # so luong sample (o day m=353 neu dung tap train)
    y_hat = w * x + b             # numpy tu dong tinh w*x+b cho MOI phan tu trong
                                   # mang x cung luc (gọi la "vectorization") - day
                                   # la vi du CU THE cho ly do ta import numpy o tren:
                                   # KHONG co numpy, dong nay phai viet bang for-loop
                                   # thu cong qua tung phan tu cua x.
    cost = np.sum((y_hat - y) ** 2) / (2 * m)   # dung DUNG cong thuc da giai thich o tren
    return cost


# Trích riêng cột bmi để train (Simple Linear Regression)
bmi_idx = feature_names.index('bmi')  # lấy index của cột bmi
x_bmi_train = X_train[:, bmi_idx]
x_bmi_test = X_test[:, bmi_idx]

cost_at_zero = compute_cost(x_bmi_train, y_train, w=14, b=6)
print(f"\nCost  {round(cost_at_zero, 2)}")





