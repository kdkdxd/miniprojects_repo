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
    test_size=0.2,             # chia ra 20% test, 80% train
    random_state=4242
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
x_bmi_train = X_train[:, bmi_idx]     # lấy cột bmi của X_train
x_bmi_test = X_test[:, bmi_idx]

cost_at_zero = compute_cost(x_bmi_train, y_train, w=0, b=0)   # Tinh MSE (Cost Function)
print(f"\nCost at w=0 and b=0 {round(cost_at_zero, 2)}")

# Số càng lớn thì Đường Hồi Quy tệ, reverse

# GRADIENT DESCENT FROM SCRATCH

def gradient_descent_1var(x, y, w_init, b_init, alpha, iterations):
    """
    Chay Gradient Descent cho Simple Linear Regression (1 feature).

    Tham so:
        x, y        : du lieu train (mang numpy 1 chieu)
        w_init      : gia tri KHOI TAO cua w (thuong bat dau tu 0)
        b_init      : gia tri KHOI TAO cua b (thuong bat dau tu 0)
        alpha       : learning rate (toc do hoc)
        iterations  : so vong lap se chay

    Tra ve:
        w, b          : gia tri w, b SAU KHI hoan tat toan bo vong lap
        cost_history  : list luu lai cost sau MOI vong lap, dung de VE
                        BIEU DO hoi tu (convergence plot) o Phan 1.6
    """
    m = len(x)
    w, b = w_init, b_init
    cost_history = []

    for i in range(iterations):
        y_hat = w * x + b
        dw = np.sum((y_hat-y) * x) / m 
        db = np.sum((y_hat-y)) / m

        w = w - alpha * dw
        b = b - alpha * db

        cost_history.append(compute_cost(x,y,w,b))

    return w, b, cost_history

# CHẠY THẬT với dữ liệu bmi (353 bệnh nhân trong tập train)
w_final, b_final, cost_history = gradient_descent_1var(
    x_bmi_train, y_train,
    w_init=0.0, b_init= 0.0,
    alpha=0.8,
    iterations=10000
)

print(f'Cost tại vòng lặp đầu tiên : {round(cost_history[0],4)}')
print(f'Cost tại vòng lặp thứ 1000 : {round(cost_history[999],4)}')
print(f"Cost tại vòng lặp cuối : {round(cost_history[9999],4)}")
print()
print(f'Kết quả cuối : w = {w_final:.4f}, b = {b_final:.4f}')


# Compare to Linear Regression
lr_bmi = LinearRegression()

lr_bmi.fit(x_bmi_train.reshape(-1,1), y_train)
print(f'Scratch Linear Regression : w = {round(w_final, 4)}, b = {round(b_final, 4)}')
print(f'Scikit Learn : w = {round(lr_bmi.coef_[0],4)}, b = {round(lr_bmi.intercept_,4)}')
    








    