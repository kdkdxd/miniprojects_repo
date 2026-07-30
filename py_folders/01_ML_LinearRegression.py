# %%
# ==============================================================================
#                    MACHINE LEARNING CHO DATA SCIENCE
#                    GIÁO TRÌNH TOÀN DIỆN - PHẦN 1/8
#                    NỀN TẢNG ML + HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)
# ==============================================================================
#
# ------------------------------------------------------------------------------
# TẠI SAO CHIA THÀNH 8 PHẦN THAY VÌ 1 FILE DUY NHẤT?
# ------------------------------------------------------------------------------
# "Toàn bộ ML" thực chất là nội dung của cả một môn học đại học (thường dạy
# trong 10-15 tuần). Nếu nhồi hết vào 1 file với mức độ annotate từng dòng,
# từng công thức, có ví dụ số cụ thể như file này đang làm, thì file đó sẽ dài
# hàng chục nghìn dòng -> hoặc bị cắt xén (mất chi t iết), hoặc chứa sai sót do
# quá tải. Cách học "brick by brick" (xây từng viên gạch chắc chắn rồi mới xây
# viên tiếp theo) sẽ hiệu quả hơn nhiều so với đọc 1 file khổng lồ.
#
# Roadmap dưới đây là bức tranh toàn cảnh của TOÀN BỘ khoá học. File này
# (Phần 1) sẽ hoàn thành 100% nội dung của PHẦN 0 và PHẦN 1. Các phần sau sẽ
# là các file riêng, học tuần tự.
#
# ------------------------------------------------------------------------------
# LỘ TRÌNH ĐẦY ĐỦ - MACHINE LEARNING CHO DATA SCIENCE
# ------------------------------------------------------------------------------
# PHẦN 0 - NỀN TẢNG & TƯ DUY ML                                    [FILE NÀY]
#   ML là gì, phân loại bài toán, quy trình làm việc, các khái niệm cốt lõi
#
# PHẦN 1 - HỒI QUY (REGRESSION)                                    [FILE NÀY]
#   Linear Regression từ lý thuyết -> from-scratch -> thư viện, Gradient
#   Descent, Normal Equation, các chỉ số đánh giá (MSE/RMSE/MAE/R2),
#   Overfitting/Underfitting, Regularization (Ridge/Lasso), Feature Scaling
#
# PHẦN 2 - PHÂN LOẠI (CLASSIFICATION)                               [file sau]
#   Logistic Regression, K-Nearest Neighbors, Decision Tree, SVM, Naive Bayes,
#   Confusion Matrix, Precision/Recall/F1, ROC-AUC
#
# PHẦN 3 - ENSEMBLE METHODS                                         [file sau]
#   Bagging, Random Forest, Boosting (AdaBoost, Gradient Boosting, XGBoost),
#   Stacking
#
# PHẦN 4 - HỌC KHÔNG GIÁM SÁT (UNSUPERVISED LEARNING)                [file sau]
#   K-Means, Hierarchical Clustering, DBSCAN, PCA, t-SNE
#
# PHẦN 5 - FEATURE ENGINEERING & DATA PREPROCESSING                  [file sau]
#   Xử lý missing data, encoding biến phân loại, feature scaling nâng cao,
#   feature selection
#
# PHẦN 6 - MODEL SELECTION & VALIDATION                              [file sau]
#   Cross-validation (K-Fold), Grid Search, Random Search, Learning Curves,
#   chẩn đoán bias/variance
#
# PHẦN 7 - NHẬP MÔN NEURAL NETWORKS                                  [file sau]
#   Perceptron, Multi-layer Perceptron, Activation Functions, Backpropagation,
#   giới thiệu CNN/RNN
#
# PHẦN 8 - CHỦ ĐỀ MỞ RỘNG                                            [file sau]
#   Imbalanced data, Time series cơ bản, Model interpretability (feature
#   importance, SHAP), giới thiệu MLOps
#
# ------------------------------------------------------------------------------
# NGUỒN THAM KHẢO (uy tín, dùng để đối chiếu lý thuyết trong file này)
# ------------------------------------------------------------------------------
#  - Stanford CS229 (Andrew Ng) - Machine Learning lecture notes
#  - Aurelien Geron - "Hands-On Machine Learning with Scikit-Learn, Keras
#    and TensorFlow" (O'Reilly)
#  - James, Witten, Hastie, Tibshirani - "An Introduction to Statistical
#    Learning" (ISLR)
#  - Tài liệu chính thức scikit-learn (scikit-learn.org)
#
# ------------------------------------------------------------------------------
# QUY TẮC VÀNG CỦA FILE NÀY
# ------------------------------------------------------------------------------
#  1. KHÔNG dùng dữ liệu giả/bịa. Toàn bộ ví dụ dùng dataset THẬT:
#     sklearn.datasets.load_diabetes() - 442 bệnh nhân tiểu đường thật, và
#     sklearn.datasets.load_breast_cancer() - 569 ca chẩn đoán ung thư vú thật.
#  2. Mọi công thức đều được suy luận từng bước, không nhảy cóc.
#  3. Mọi khái niệm đều có ví dụ số cụ thể (không chỉ nói suông lý thuyết).
#  4. Mọi đoạn code đều được giải thích: KHÔNG dùng nó thì sao (break),
#     xuất hiện ở đâu (context), số liệu cụ thể (concrete), so sánh cách khác
#     (compare), và một câu tổng kết (summary).
#  5. Chạy file này bằng VS Code Interactive Window: bấm vào từng khối bắt đầu
#     bằng "# %%" và nhấn Shift+Enter (hoặc "Run Cell") để chạy từng ô một,
#     giống hệt Jupyter Notebook nhưng vẫn là file .py thuần.
# ==============================================================================


# %%
# ==============================================================================
# PHẦN 0.1 - MACHINE LEARNING LÀ GÌ?
# ==============================================================================
#
# ĐỊNH NGHĨA (theo Tom Mitchell, được trích trong CS229):
# "Một chương trình máy tính được gọi là HỌC từ kinh nghiệm E, đối với một lớp
# nhiệm vụ T và một độ đo hiệu suất P, nếu hiệu suất của nó trên T (đo bằng P)
# CẢI THIỆN nhờ kinh nghiệm E."
#
# Nghe trừu tượng, hãy cụ thể hoá bằng đúng bài toán ta sắp làm trong file này:
#   - T (Task/nhiệm vụ)      : dự đoán mức độ tiến triển bệnh tiểu đường của
#                               một bệnh nhân, dựa trên các chỉ số y tế của họ
#   - E (Experience/kinh nghiệm): 353 hồ sơ bệnh nhân CÓ SẴN kết quả thật
#                               (dữ liệu "training")
#   - P (Performance/hiệu suất): sai số giữa dự đoán và kết quả thật (ta sẽ
#                               định nghĩa chính xác ở Phần 1.13 - MSE)
#
# ------------------------------------------------------------------------------
# SO SÁNH: LẬP TRÌNH TRUYỀN THỐNG vs MACHINE LEARNING
# ------------------------------------------------------------------------------
# Lập trình truyền thống:
#     Input + RULES (do con người viết)  --->  Output
#     Ví dụ: if bmi > 30: risk = "cao"   (con người tự nghĩ ra ngưỡng 30)
#
# Machine Learning:
#     Input + Output (dữ liệu có sẵn)    --->  RULES (máy tự "học" ra)
#     Ví dụ: đưa cho máy 353 cặp (chỉ số y tế, mức độ bệnh thật), máy tự tìm
#     ra công thức (không phải do ta đoán ngưỡng) để dự đoán mức độ bệnh cho
#     bệnh nhân MỚI mà nó chưa từng thấy.
#
# ĐIỀU GÌ "VỠ" (break) NẾU KHÔNG CÓ TƯ DUY NÀY?
# Nếu cố lập trình "truyền thống" cho bài toán dự đoán bệnh tiểu đường, ta sẽ
# phải tự đoán qua bao nhiêu chỉ số, trọng số mỗi chỉ số là bao nhiêu -> gần
# như bất khả thi vì có tới 10 chỉ số y tế tương tác phức tạp với nhau. ML học
# trọng số tối ưu tự động từ dữ liệu thật, thay vì con người đoán mò.
#
# TÓM TẮT 1 CÂU: Machine Learning là cách để MÁY TỰ TÌM RA quy luật/công thức
# từ dữ liệu có sẵn, thay vì con người viết sẵn quy luật đó.


# %%
# ==============================================================================
# PHẦN 0.2 - PHÂN LOẠI BÀI TOÁN MACHINE LEARNING
# ==============================================================================
#
# Có 3 nhóm bài toán ML chính, phân biệt dựa trên: dữ liệu ta có gồm những gì.
#
# ------------------------------------------------------------------------------
# (A) SUPERVISED LEARNING (Học có giám sát)
# ------------------------------------------------------------------------------
# Dữ liệu có cả INPUT (X - đặc trưng/features) VÀ OUTPUT ĐÚNG (y - nhãn/label).
# Máy học ánh xạ f: X -> y bằng cách so sánh dự đoán của nó với y thật, rồi
# tự sửa sai (đây là cơ chế "giám sát" - có đáp án đúng để đối chiếu).
# Supervised Learning chia làm 2 loại nhỏ, phân biệt bởi KIỂU DỮ LIỆU của y:
#
#   -> REGRESSION (Hồi quy): y là số THỰC LIÊN TỤC.
#      Ví dụ trong file này: y = mức độ tiến triển bệnh tiểu đường, một số
#      thực bất kỳ (25.0, 346.0, 87.3, ...). Đây là chủ đề CHÍNH của Phần 1.
#
#   -> CLASSIFICATION (Phân loại): y là một trong số HỮU HẠN các NHÃN/LỚP.
#      Ví dụ: y = "malignant" (ác tính) hoặc "benign" (lành tính) - chỉ có
#      2 khả năng. Đây là chủ đề của Phần 2 (file sau), nhưng file này SẼ
#      xem trước bộ dữ liệu đó ở Phần 1.17 để minh hoạ Feature Scaling.
#
# ------------------------------------------------------------------------------
# (B) UNSUPERVISED LEARNING (Học không giám sát)
# ------------------------------------------------------------------------------
# Dữ liệu CHỈ CÓ INPUT (X), KHÔNG CÓ đáp án đúng y. Máy tự tìm cấu trúc/pattern
# ẩn trong dữ liệu. Ví dụ: nhóm 1000 khách hàng thành các "cụm" có hành vi mua
# sắm giống nhau, mà không ai bảo trước "khách hàng A thuộc nhóm nào" - máy tự
# khám phá ra các nhóm. Đây là chủ đề của Phần 4.
#
# ------------------------------------------------------------------------------
# (C) REINFORCEMENT LEARNING (Học tăng cường) - chỉ giới thiệu sơ lược
# ------------------------------------------------------------------------------
# Máy (gọi là "agent") tương tác với môi trường, nhận "phần thưởng" (reward)
# hoặc "hình phạt" (penalty) sau mỗi hành động, rồi học cách tối đa hoá phần
# thưởng lâu dài. Ví dụ: AI chơi cờ vây (AlphaGo), robot học đi. Nhóm này ít
# dùng trong Data Science ứng dụng thông thường (nhiều hơn trong game/robotics)
# nên file này KHÔNG đi sâu.
#
# ------------------------------------------------------------------------------
# BẢNG SO SÁNH NHANH
# ------------------------------------------------------------------------------
#  Loại              | Có nhãn y?  | Kiểu y          | Ví dụ trong file
#  ------------------|-------------|-----------------|---------------------
#  Regression         | Có          | Số thực liên tục| Dự đoán mức độ bệnh
#  Classification      | Có          | Nhãn rời rạc    | Ác tính/lành tính
#  Unsupervised        | Không       | (không có)      | Phân cụm khách hàng
#
# TÓM TẮT 1 CÂU: Phân loại bài toán ML dựa vào việc dữ liệu CÓ đáp án đúng hay
# không, và nếu có thì đáp án đó là SỐ (regression) hay NHÃN (classification).


# %%
# ==============================================================================
# PHẦN 0.3 - QUY TRÌNH LÀM VIỆC ML (ML WORKFLOW)
# ==============================================================================
#
# Một dự án ML thực tế (kể cả trong công việc Data Scientist sau này) luôn đi
# qua các bước sau. File này sẽ SỐNG qua đúng quy trình này với dữ liệu thật.
#
#   Bước 1. THU THẬP DỮ LIỆU (Data Collection)
#           -> File này: load_diabetes() từ sklearn (dữ liệu y tế thật)
#   Bước 2. KHÁM PHÁ DỮ LIỆU (EDA - Exploratory Data Analysis)
#           -> Phần 1.1: xem shape, thống kê mô tả, tương quan
#   Bước 3. TIỀN XỬ LÝ (Preprocessing)
#           -> Phần 1.2: chia train/test; Phần 1.17: feature scaling
#   Bước 4. CHỌN MÔ HÌNH (Model Selection)
#           -> Phần 1.3 trở đi: chọn Linear Regression cho bài toán regression
#   Bước 5. HUẤN LUYỆN (Training)
#           -> Phần 1.5-1.10: Gradient Descent, Normal Equation
#   Bước 6. ĐÁNH GIÁ (Evaluation)
#           -> Phần 1.13: MSE, RMSE, MAE, R2 trên tập test
#   Bước 7. TINH CHỈNH (Tuning) / TRIỂN KHAI (Deployment)
#           -> Phần 1.15-1.16: Regularization để giảm overfitting
#           -> (Deployment thực tế nằm ngoài phạm vi giáo trình này)
#
# ĐIỀU GÌ "VỠ" NẾU BỎ QUA MỘT BƯỚC?
#   - Bỏ qua EDA -> có thể không phát hiện dữ liệu bị lỗi, thiếu, hoặc các đặc
#     trưng không liên quan gì đến y -> lãng phí công sức huấn luyện mô hình
#     trên dữ liệu "rác".
#   - Bỏ qua chia train/test (Bước 3) -> không có cách nào biết mô hình có
#     "học vẹt" (overfitting) hay không, vì đánh giá trên chính dữ liệu đã
#     học sẽ luôn cho kết quả đẹp giả tạo.
#   - Bỏ qua Bước 7 (tinh chỉnh) -> mô hình có thể hoạt động tệ trên dữ liệu
#     mới dù rất tốt trên dữ liệu huấn luyện (sẽ thấy rõ ở Phần 1.14).
#
# TÓM TẮT 1 CÂU: ML không phải là "gọi 1 hàm .fit() là xong" - nó là một QUY
# TRÌNH nhiều bước, và file này sẽ đi qua đủ cả 7 bước với dữ liệu thật.


# %%
# ==============================================================================
# PHẦN 0.4 - CÁC KHÁI NIỆM CỐT LÕI (phải thuộc lòng trước khi học tiếp)
# ==============================================================================
#
# Những từ này sẽ xuất hiện SUỐT toàn bộ giáo trình 8 phần, không riêng gì
# Linear Regression, nên cần hiểu chắc ngay từ đầu.
#
#  - FEATURE (đặc trưng, ký hiệu X, hoặc x1, x2...xn):
#    Các thông tin đầu vào dùng để dự đoán. Trong dataset của ta: age, sex,
#    bmi, bp, s1...s6 (10 features).
#
#  - LABEL / TARGET (nhãn/mục tiêu, ký hiệu y):
#    Giá trị ta muốn dự đoán. Trong dataset của ta: mức độ tiến triển bệnh
#    (1 số thực cho mỗi bệnh nhân).
#
#  - SAMPLE / INSTANCE (mẫu/quan sát):
#    Một "dòng" dữ liệu, ví dụ: 1 bệnh nhân cụ thể với đủ 10 chỉ số + 1 target.
#    Dataset của ta có 442 samples.
#
#  - MODEL (mô hình):
#    Một hàm số f(X) = y_hat (đọc là "y mũ", ký hiệu dự đoán, phân biệt với y
#    thật). Với Linear Regression, model chính là công thức
#    y_hat = w1*x1 + w2*x2 + ... + wn*xn + b (sẽ giải thích kỹ ở Phần 1.3).
#
#  - PARAMETER (tham số, vd: w và b ở trên):
#    Các con số BÊN TRONG model mà MÁY TỰ HỌC ra từ dữ liệu (ta không tự đặt).
#
#  - HYPERPARAMETER (siêu tham số, vd: learning rate alpha ở Phần 1.5):
#    Các con số CON NGƯỜI tự chọn TRƯỚC khi huấn luyện (máy không tự học ra).
#    Phân biệt: "parameter" máy tự tìm, "hyperparameter" người tự chọn.
#
#  - TRAIN SET / TEST SET (tập huấn luyện / tập kiểm tra):
#    Chia dữ liệu thành 2 phần KHÔNG GIAO NHAU: train để máy HỌC, test để
#    ĐÁNH GIÁ xem máy học tốt đến đâu trên dữ liệu nó CHƯA TỪNG THẤY.
#    (Chi tiết + code thật ở Phần 1.2)
#
#  - OVERFITTING (học vẹt / học quá khớp):
#    Mô hình "học thuộc lòng" luôn cả nhiễu (noise) trong tập train, nên làm
#    RẤT TỐT trên train nhưng làm TỆ trên test/dữ liệu mới.
#    Ẩn dụ: học sinh học thuộc lòng đáp án của ĐÚNG 10 câu trong đề cương,
#    thi đạt 10/10 nếu đề thi ra ĐÚNG 10 câu đó, nhưng 0 điểm nếu đổi số liệu.
#
#  - UNDERFITTING (học chưa đủ / học quá đơn giản):
#    Mô hình QUÁ ĐƠN GIẢN, không nắm được cả quy luật CƠ BẢN trong dữ liệu,
#    nên làm TỆ cả trên train LẪN test.
#    Ẩn dụ: học sinh chỉ học mỗi phép cộng nhưng đề thi có cả nhân chia -> sai
#    ngay từ khi làm bài tập mẫu (train), chứ chưa nói đến đề thi thật (test).
#
#  - BIAS - VARIANCE TRADEOFF (đánh đổi giữa độ chệch và phương sai):
#    BIAS cao <=> mô hình quá đơn giản, giả định sai lệch nhiều so với thực tế
#    -> underfitting.
#    VARIANCE cao <=> mô hình quá nhạy với từng biến động nhỏ của tập train
#    -> overfitting.
#    Mục tiêu của việc "tuning" một mô hình là tìm điểm CÂN BẰNG giữa 2 thái
#    cực này - sẽ thấy rất rõ qua ví dụ số cụ thể ở Phần 1.14.
#
# TÓM TẮT 1 CÂU: Feature/label là đầu vào/đầu ra; parameter máy tự học còn
# hyperparameter người tự chọn; và mục tiêu tối thượng của ML là mô hình
# KHÔNG học vẹt (overfit) cũng KHÔNG học hời hợt (underfit).


# %%
# ==============================================================================
# PHẦN 1.0 - GIỚI THIỆU BÀI TOÁN HỒI QUY + NẠP DỮ LIỆU THẬT
# ==============================================================================
#
# Từ đây, ta bắt đầu PHẦN 1: HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION) - thuật
# toán ML cổ điển nhất, và cũng là NỀN TẢNG để hiểu rất nhiều thuật toán phức
# tạp hơn sau này (Logistic Regression ở Phần 2 chỉ là Linear Regression cộng
# thêm 1 hàm biến đổi; Neural Network ở Phần 7 về bản chất là NHIỀU lớp Linear
# Regression xếp chồng lên nhau). Hiểu THẬT SÂU phần này sẽ giúp mọi phần sau
# dễ hơn rất nhiều.
#
# BÀI TOÁN: Cho các chỉ số y tế của một bệnh nhân tiểu đường (tuổi, giới tính,
# BMI, huyết áp, 6 chỉ số huyết thanh), hãy dự đoán một con số đo mức độ tiến
# triển của bệnh sau 1 năm.
#
# DATASET DÙNG TRONG TOÀN BỘ PHẦN 1: "Diabetes dataset" - dữ liệu THẬT của
# 442 bệnh nhân tiểu đường, được thu thập bởi Efron, Hastie, Johnstone,
# Tibshirani (2004, bài báo "Least Angle Regression", Annals of Statistics),
# và được đóng gói SẴN bên trong thư viện scikit-learn (không cần tải mạng,
# không có rủi ro dữ liệu bị lỗi khi tải) - vì vậy đây là lựa chọn ĐÁNG TIN
# CẬY để học, dù là dữ liệu THẬT 100%, không phải dữ liệu tự bịa.

import numpy as np                              # tính toán số học, ma trận, vector
import pandas as pd                              # xử lý dữ liệu dạng bảng (DataFrame)
import matplotlib.pyplot as plt                  # vẽ biểu đồ
import seaborn as sns                            # vẽ biểu đồ thống kê đẹp hơn (dựa trên matplotlib)
from sklearn.datasets import load_diabetes, load_breast_cancer   # 2 dataset THẬT dùng trong file
from sklearn.model_selection import train_test_split             # chia train/test
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # các mô hình hồi quy có sẵn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # tiền xử lý
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # đánh giá

# GIẢI THÍCH TỪNG DÒNG IMPORT (tại sao cần, KHÔNG có thì sao):
#
# - numpy (viết tắt "np"): là thư viện TÍNH TOÁN SỐ HỌC nền tảng của Python.
#   KHÔNG có numpy: mọi phép toán vector/ma trận (như w1*x1+w2*x2+...) phải
#   viết bằng vòng lặp for thủ công -> chậm hơn HÀNG TRĂM LẦN và code dài dòng.
#   numpy cho phép viết "X.dot(w)" thay vì for-loop cộng dồn từng phần tử.
#
# - pandas (viết tắt "pd"): thư viện xử lý dữ liệu DẠNG BẢNG (giống Excel).
#   KHÔNG có pandas: dữ liệu chỉ là mảng numpy "vô danh" (không có tên cột),
#   rất khó xem/lọc/thống kê theo từng đặc trưng cụ thể (vd: "cho tôi xem cột
#   bmi") - đây đúng là phần bạn đã luyện tập rất nhiều trong các buổi học
#   pandas trước, giờ áp dụng lại vào ML.
#
# - matplotlib.pyplot (viết tắt "plt"): thư viện VẼ BIỂU ĐỒ cơ bản của Python.
#   KHÔNG có nó: không thể "nhìn thấy" dữ liệu/mô hình -> rất khó phát hiện
#   overfitting bằng mắt thường (sẽ thấy rõ ở Phần 1.14).
#
# - seaborn (viết tắt "sns"): xây dựng TRÊN NỀN matplotlib, chuyên vẽ biểu đồ
#   THỐNG KÊ (heatmap tương quan, phân phối...) với code ngắn gọn hơn.
#
# - sklearn (scikit-learn): thư viện ML chuẩn công nghiệp của Python, chứa:
#     + datasets: các dataset mẫu (load_diabetes, load_breast_cancer)
#     + model_selection: công cụ chia dữ liệu, cross-validation
#     + linear_model: các mô hình hồi quy tuyến tính đã tối ưu sẵn
#     + preprocessing: các công cụ tiền xử lý (chuẩn hoá, tạo đặc trưng đa
#       thức...)
#     + metrics: các công thức đánh giá mô hình đã viết sẵn, dùng để ĐỐI
#       CHIẾU với code "from scratch" (tự viết tay) mà ta sẽ xây dựng bên
#       dưới - đây là cách kiểm tra code tự viết có ĐÚNG hay không.
#
# TẠI SAO FILE NÀY LUÔN TỰ VIẾT (from scratch) RỒI MỚI DÙNG THƯ VIỆN?
# Vì nếu chỉ gọi model.fit() ngay từ đầu, bạn sẽ dùng được ML như một "hộp
# đen" (black box) nhưng KHÔNG hiểu bên trong nó làm gì -> không thể debug khi
# mô hình dự đoán sai, không thể tuỳ biến khi cần, và không hiểu vì sao các
# thuật toán phức tạp hơn (Phần 2 trở đi) lại hoạt động như vậy. File này XÂY
# TỪ ĐẦU trước, để hiểu bản chất, RỒI mới dùng thư viện (nhanh, tối ưu, dùng
# trong công việc thật) và ĐỐI CHIẾU 2 kết quả để chắc chắn hiểu đúng.

# %%
# Nạp dataset THẬT
data = load_diabetes()
X = data.data          # ma trận đặc trưng (features), shape (442, 10)
y = data.target        # vector nhãn (target/label), shape (442,)
feature_names = list(data.feature_names)

print("Shape của X (features):", X.shape)
print("Shape của y (target):", y.shape)
print("Tên 10 đặc trưng:", feature_names)
print()
print("Dòng dữ liệu đầu tiên (bệnh nhân số 0), X[0]:")
print(X[0])
print("Target thật của bệnh nhân số 0, y[0]:", y[0])

# GIẢI THÍCH KẾT QUẢ IN RA (con số CỤ THỂ, không phải lý thuyết suông):
#
# "Shape của X: (442, 10)" nghĩa là: 442 SAMPLES (bệnh nhân) x 10 FEATURES
# (chỉ số y tế). Đây CHÍNH LÀ khái niệm "sample" và "feature" đã học ở Phần
# 0.4, giờ nhìn thấy bằng số liệu thật.
#
# X[0] = [0.038076  0.05068   0.061696  0.021872 -0.044223 -0.034821 -0.043401
#        -0.002592  0.019907 -0.017646]
# Đây là 10 chỉ số y tế của bệnh nhân đầu tiên, THEO ĐÚNG THỨ TỰ feature_names
# = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']. Tức là bệnh nhân
# này có age=0.038076, sex=0.05068, bmi=0.061696, ...
#
# ĐIỂM ĐẶC BIỆT PHẢI LƯU Ý: các con số này KHÔNG PHẢI đơn vị gốc (không phải
# "tuổi 45", "BMI 28.3"...). scikit-learn đã CHỦ ĐỘNG chuẩn hoá sẵn (center +
# scale) 10 cột này trước khi đóng gói - ta sẽ KIỂM CHỨNG điều này bằng số
# liệu thật ngay bên dưới. Đây VẪN LÀ dữ liệu thật (không phải bịa), chỉ là
# đơn vị đo đã được biến đổi. Đây thực ra là 1 ví dụ THẬT của khái niệm
# "Feature Scaling" mà ta sẽ học kỹ ở Phần 1.17 - dataset này đã làm sẵn cho
# ta xem trước!
#
# y[0] = 151.0: đây LÀ đơn vị gốc, chưa bị biến đổi - là chỉ số đo mức độ
# tiến triển bệnh tiểu đường của bệnh nhân 0, sau 1 năm theo dõi.

# %%
# ==============================================================================
# PHẦN 1.1 - EDA (EXPLORATORY DATA ANALYSIS) - KHÁM PHÁ DỮ LIỆU
# ==============================================================================
#
# Trước khi đưa dữ liệu vào bất kỳ mô hình nào, LUÔN LUÔN phải "nhìn" dữ liệu
# trước - đây chính là Bước 2 trong ML Workflow (Phần 0.3). Ta sẽ dùng lại
# đúng kỹ năng pandas EDA đã luyện tập trước đây (loc/iloc, describe, corr...)

# đưa dữ liệu numpy vào DataFrame để dễ thao tác (đây là lý do ta học pandas
# trước ML: pandas là công cụ THAO TÁC dữ liệu, sklearn là công cụ HỌC MÔ HÌNH)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("5 dòng đầu:")
print(df.head())
print()
print("Thống kê mô tả (describe) - full 11 cột (10 features + target):")
print(df.describe().round(4))

# GIẢI THÍCH KẾT QUẢ (kiểm chứng điều đã nói ở Phần 1.0 bằng số liệu THẬT):
#
# Nhìn hàng "mean" của 10 cột feature: TẤT CẢ đều xấp xỉ 0.0000 (age, sex,
# bmi... đều ~0). Nhìn hàng "std": TẤT CẢ đều CHÍNH XÁC bằng 0.0476. Đây
# không phải trùng hợp - đây là BẰNG CHỨNG SỐ HỌC cho việc scikit-learn đã
# "center" (trừ đi trung bình, nên mean=0) và "scale" (chia cho 1 hằng số
# chung, nên std bằng nhau ở MỌI cột) cho cả 10 features TRƯỚC khi đóng gói.
#
# Ngược lại, cột "target" có mean=152.13, std=77.09, min=25, max=346 - đây
# LÀ đơn vị đo gốc, KHÔNG bị biến đổi gì cả.

# %%
# Tính tương quan (correlation) của từng feature với target - đây là bước
# quan trọng để biết feature nào "liên quan" nhiều nhất đến điều ta muốn dự
# đoán, TRƯỚC KHI huấn luyện bất kỳ mô hình nào.
correlation_with_target = df.corr()['target'].sort_values(ascending=False)
print("Hệ số tương quan (correlation) giữa từng feature và target:")
print(correlation_with_target.round(4))

# GIẢI THÍCH Ý NGHĨA CỦA HỆ SỐ TƯƠNG QUAN (Pearson correlation, ký hiệu r):
# r nằm trong khoảng [-1, 1].
#   r gần +1  -> feature và target CÙNG TĂNG CÙNG GIẢM (tương quan thuận)
#   r gần -1  -> feature tăng thì target GIẢM, và ngược lại (tương quan nghịch)
#   r gần  0  -> hầu như KHÔNG có quan hệ tuyến tính giữa 2 đại lượng
#
# Đọc kết quả THẬT ở trên:
#   bmi:  r = 0.5865  -> tương quan thuận MẠNH NHẤT: BMI càng cao, mức độ
#                        tiến triển bệnh càng có xu hướng cao -> hợp lý về
#                        mặt y khoa (béo phì là yếu tố nguy cơ tiểu đường)
#   s5:   r = 0.5659  -> tương quan thuận mạnh thứ nhì
#   s3:   r = -0.3948 -> tương quan NGHỊCH: s3 (một chỉ số huyết thanh) càng
#                        cao thì mức độ bệnh có xu hướng càng THẤP
#   sex:  r = 0.0431  -> gần 0, hầu như KHÔNG liên quan tuyến tính đến target
#
# CHÍNH VÌ bmi có tương quan mạnh nhất, ta sẽ dùng feature "bmi" làm ví dụ
# XUYÊN SUỐT cho phần Hồi quy tuyến tính ĐƠN BIẾN (chỉ 1 feature) ở các phần
# 1.3 - 1.8 bên dưới - chọn feature có tương quan mạnh sẽ giúp nhìn thấy rõ
# đường hồi quy "khớp" với dữ liệu như thế nào trên biểu đồ.
#
# ĐIỀU GÌ "VỠ" NẾU BỎ QUA BƯỚC NÀY? Nếu không xem tương quan trước, ta có thể
# vô tình chọn 1 feature GẦN NHƯ KHÔNG liên quan (vd: "sex", r=0.0431) để
# minh hoạ, và sẽ thấy đường hồi quy gần như nằm ngang, KHÔNG khớp gì với dữ
# liệu -> hiểu lầm rằng Linear Regression "không hoạt động", trong khi vấn đề
# thực ra là chọn sai feature để minh hoạ.

# %%
# Trực quan hoá ma trận tương quan bằng heatmap (bản đồ nhiệt)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Ma tran tuong quan giua 10 dac trung va target (Diabetes dataset that)')
plt.tight_layout()
plt.show()

# GIẢI THÍCH: mỗi ô (i, j) trong heatmap là hệ số tương quan r giữa feature i
# và feature j. Màu ĐỎ ĐẬM = tương quan thuận mạnh (gần +1), màu XANH ĐẬM =
# tương quan nghịch mạnh (gần -1), màu NHẠT = gần 0 (không liên quan).
#
# QUAN SÁT QUAN TRỌNG cho các phần sau: hãy để ý ô giao giữa "s1" và "s2" -
# 2 cột này có màu ĐỎ RẤT ĐẬM (tương quan rất cao). Đây CHÍNH LÀ điều sẽ gây
# ra hiện tượng thú vị ở Phần 1.11 (Gradient Descent hội tụ chậm khi các
# feature tương quan mạnh với nhau - gọi là "multicollinearity"/đa cộng
# tuyến) - cứ ghi nhớ quan sát này, ta sẽ quay lại giải thích kỹ.


# %%
# ==============================================================================
# PHẦN 1.2 - CHIA TẬP TRAIN/TEST
# ==============================================================================
#
# TẠI SAO PHẢI CHIA? (đã nói lý thuyết ở Phần 0.3, giờ làm THẬT với code)
# Nếu huấn luyện mô hình trên TOÀN BỘ 442 bệnh nhân, rồi cũng ĐÁNH GIÁ mô hình
# trên chính 442 bệnh nhân đó, thì kết quả đánh giá sẽ LUÔN đẹp giả tạo -
# giống hệt việc học sinh "học tủ" đúng 10 câu rồi thi lại đúng 10 câu đó.
# Ta cần một phần dữ liệu HOÀN TOÀN TÁCH BIỆT mà mô hình CHƯA TỪNG "nhìn thấy"
# trong lúc học, để đánh giá xem nó có thực sự "hiểu" quy luật hay chỉ "học
# vẹt" (overfitting - đã định nghĩa ở Phần 0.4).

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # dành 20% dữ liệu cho tập test, 80% cho tập train
    random_state=42     # "hạt giống" ngẫu nhiên cố định, để MỖI LẦN chạy lại
                         # file này đều cho ra CÙNG MỘT cách chia - giúp kết
                         # quả có thể tái lập (reproducible), rất quan trọng
                         # khi debug hoặc so sánh các mô hình với nhau.
)

print("Kich thuoc X_train:", X_train.shape)
print("Kich thuoc X_test:", X_test.shape)
print("Kich thuoc y_train:", y_train.shape)
print("Kich thuoc y_test:", y_test.shape)
print()
print("5 gia tri dau cua y_train:", y_train[:5])

# GIẢI THÍCH CON SỐ THẬT:
# X_train.shape = (353, 10)  -> 353 bệnh nhân (=80% của 442, làm tròn) dùng
#                                để HUẤN LUYỆN
# X_test.shape  = (89, 10)   -> 89 bệnh nhân (=20% của 442) dùng để ĐÁNH GIÁ,
#                                mô hình sẽ KHÔNG BAO GIỜ được "nhìn thấy" 89
#                                bệnh nhân này trong lúc học (Phần 1.5-1.12)
# 353 + 89 = 442 -> đúng bằng tổng số bệnh nhân ban đầu, không mất/thêm dữ
# liệu, chỉ là CHIA lại.
#
# random_state=42: đây là quy ước phổ biến trong cộng đồng ML (không có ý
# nghĩa đặc biệt, 42 chỉ là 1 con số quen thuộc được dùng làm ví dụ - nhưng
# QUAN TRỌNG là con số này PHẢI CỐ ĐỊNH xuyên suốt để mọi so sánh sau này
# (Gradient Descent vs Normal Equation vs sklearn) đều dùng CÙNG MỘT cách
# chia dữ liệu, nếu không sẽ so sánh "táo với cam".
#
# SO SÁNH: NẾU KHÔNG DÙNG random_state (để mặc định ngẫu nhiên)
#   -> Mỗi lần chạy lại file sẽ chia dữ liệu KHÁC NHAU -> kết quả (MSE, R2...)
#      sẽ thay đổi mỗi lần chạy -> không thể so sánh "cải thiện" hay "tệ đi"
#      một cách công bằng giữa các lần thử nghiệm khác nhau.
#
# TÓM TẮT 1 CÂU: train_test_split tách 442 bệnh nhân thành 353 để HỌC và 89
# để KIỂM TRA - và random_state=42 đảm bảo cách tách này LUÔN GIỐNG NHAU mỗi
# lần chạy lại.

# %%
# ==============================================================================
# PHẦN 1.3 - LY THUYET: HOI QUY TUYEN TINH DON BIEN (SIMPLE LINEAR REGRESSION)
# ==============================================================================
#
# Ta bắt đầu với trường hợp ĐƠN GIẢN NHẤT: chỉ dùng 1 feature (bmi) để dự
# đoán target. Sau khi hiểu chắc trường hợp 1 feature, việc mở rộng ra 10
# features (Phần 1.9) sẽ chỉ là một bước tự nhiên, không có gì khó thêm về
# mặt Ý TƯỞNG (chỉ khó hơn về mặt TÍNH TOÁN, và numpy sẽ lo phần đó cho ta).
#
# Ý TƯỞNG HÌNH HỌC: nếu vẽ bmi trên trục hoành (x) và target trên trục tung
# (y), mỗi bệnh nhân là 1 điểm chấm trên mặt phẳng. Linear Regression đi tìm
# ĐƯỜNG THẲNG "khớp" nhất với các điểm chấm đó, để sau này với 1 bmi MỚI
# (bệnh nhân mới), ta chỉ cần nhìn lên đường thẳng đó để đọc ra target dự
# đoán, thay vì phải có sẵn dữ liệu thật.
#
# CÔNG THỨC (phương trình đường thẳng, chắc bạn đã quen từ Toán phổ thông):
#
#         y_hat = w * x + b
#
#   - x       : giá trị feature đầu vào (ở đây là bmi của 1 bệnh nhân)
#   - y_hat   : giá trị DỰ ĐOÁN (đọc là "y mu") - phân biệt với y THẬT
#   - w       : "weight" / hệ số góc (độ dốc) của đường thẳng - quyết định
#               x tăng 1 đơn vị thì y_hat thay đổi bao nhiêu
#   - b       : "bias" / hệ số chặn (giao điểm với trục tung khi x=0)
#
# ĐÂY CHÍNH LÀ 2 "PARAMETER" đã nhắc ở Phần 0.4 - máy sẽ TỰ TÌM ra giá trị
# w và b tốt nhất từ 353 bệnh nhân trong tập train, KHÔNG PHẢI ta tự đoán.
#
# CÂU HỎI CỐT LÕI: có VÔ SỐ đường thẳng (w, b) có thể vẽ qua mặt phẳng đó -
# làm sao biết đường nào là "tốt nhất"? Ta cần một cách ĐO LƯỜNG "độ tệ" của
# một đường thẳng cụ thể -> đó chính là HÀM MẤT MÁT (Cost Function), học ở
# Phần 1.4 ngay sau đây.


# %%
# ==============================================================================
# PHẦN 1.4 - HAM MAT MAT (COST FUNCTION) - MEAN SQUARED ERROR (MSE)
# ==============================================================================
#
# Ý TƯỞNG: với một cặp (w, b) bất kỳ, ta tính y_hat cho TỪNG bệnh nhân trong
# tập train, rồi so sánh với y THẬT của họ. Nếu y_hat gần y thật -> đường
# thẳng "tốt". Nếu y_hat lệch xa y thật -> đường thẳng "tệ". Hàm mất mát
# LƯỢNG HÓA "độ tệ" này thành 1 con số duy nhất.
#
# CÔNG THỨC MSE (Mean Squared Error - sai số bình phương trung bình):
#
#         J(w, b) = (1 / 2m) * SUM[i=1 den m] (y_hat_i - y_i)^2
#
#   trong do y_hat_i = w * x_i + b, va m = so luong sample (o day m = 353,
#   so benh nhan trong tap train).
#
# GIẢI THÍCH TỪNG THÀNH PHẦN CỦA CÔNG THỨC (lời văn, từng bước):
#
#  1) (y_hat_i - y_i): đây là SAI SỐ (error/residual) của RIÊNG bệnh nhân i -
#     dự đoán trừ đi giá trị thật. Nếu dự đoán CAO hơn thật -> số dương. Nếu
#     dự đoán THẤP hơn thật -> số âm.
#
#  2) TẠI SAO BÌNH PHƯƠNG (^2) sai số, thay vì chỉ cộng thẳng sai số lại?
#     Nếu KHÔNG bình phương mà cộng thẳng các (y_hat_i - y_i) lại, một mô
#     hình dự đoán bệnh nhân A cao hơn 100 và bệnh nhân B thấp hơn 100 sẽ có
#     TỔNG SAI SỐ = 0 (100 + (-100) = 0) -> trông có vẻ "hoàn hảo" dù thực ra
#     sai rất nhiều ở CẢ HAI bệnh nhân! Bình phương biến MỌI sai số (dù âm
#     hay dương) thành số DƯƠNG, nên sai số không thể "triệt tiêu nhau" theo
#     kiểu giả tạo đó. Bình phương còn có tác dụng PHẠT NẶNG các sai số LỚN
#     hơn nhiều so với sai số nhỏ (sai 10 bị phạt 100, sai 20 bị phạt 400 -
#     gấp 4 lần chứ không phải gấp 2) - đây là tính chất TA MUỐN, vì một dự
#     đoán sai lệch RẤT NHIỀU thường nguy hiểm hơn nhiều dự đoán sai lệch ít.
#
#  3) SUM[i=1 den m]: cộng dồn bình phương sai số của TẤT CẢ m bệnh nhân lại,
#     để có một con số đại diện cho "độ tệ" của TOÀN BỘ tập train, chứ không
#     chỉ riêng 1 bệnh nhân.
#
#  4) CHIA CHO m (lấy TRUNG BÌNH): để con số J không phụ thuộc vào việc tập
#     train có bao nhiêu bệnh nhân. Nếu KHÔNG chia cho m, một tập có 1000
#     bệnh nhân sẽ luôn có J LỚN HƠN một tập có 100 bệnh nhân dù MÔ HÌNH
#     THỰC RA TỐT NGANG NHAU - chia cho m giúp so sánh công bằng.
#
#  5) CHIA CHO 2 (chứ không chỉ chia cho m): đây LÀ MẸO TOÁN HỌC THUẦN TUÝ,
#     không làm thay đổi đường thẳng (w, b) tối ưu tìm được (vì nhân J với 1
#     hằng số dương không đổi vị trí điểm cực tiểu). Lý do có số 2 này: khi
#     lấy đạo hàm của (...)^2 ở Phần 1.5 ngay dưới đây, số mũ 2 sẽ "rơi
#     xuống" nhân vào biểu thức (theo quy tắc đạo hàm hàm hợp), và số 2 đó sẽ
#     TRIỆT TIÊU đúng với số 2 ở mẫu số -> công thức đạo hàm cuối cùng gọn
#     gàng hơn, không có hệ số 2 thừa. Bạn sẽ THẤY rõ điều này ở Phần 1.5.

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

# VÍ DỤ CỤ THỂ với con số THẬT: thử với đường thẳng "vô nghĩa" nhất, w=0,
# b=0 (nghĩa là dự đoán MỌI bệnh nhân đều có target = 0, hiển nhiên rất tệ):
bmi_idx = feature_names.index('bmi')
x_bmi_train = X_train[:, bmi_idx]     # trich RIENG cot bmi tu X_train, shape (353,)
x_bmi_test = X_test[:, bmi_idx]       # trich RIENG cot bmi tu X_test, shape (89,), 
                                       # se dung sau nay de danh gia

cost_at_zero = compute_cost(x_bmi_train, y_train, w=0, b=0)
print("Cost khi w=0, b=0 (du doan tat ca deu bang 0):", round(cost_at_zero, 2))

# GIẢI THÍCH KẾT QUẢ THẬT: cost = 14855.66 - đây là con số RẤT LỚN, đúng như
# dự đoán, vì dự đoán "mọi bệnh nhân target=0" trong khi target thật trung
# bình là 152.13 (đã thấy ở Phần 1.1) -> sai số trung bình cỡ 152, bình
# phương lên thành hơn 23000, rồi chia 2 còn khoảng 11500-14855 (con số
# CHÍNH XÁC phụ thuộc cả phương sai của target, không chỉ trung bình).
# CON SỐ 14855.66 NÀY TỰ NÓ không có ý nghĩa "tốt" hay "xấu" - nó chỉ có ý
# nghĩa khi SO SÁNH với cost của một đường thẳng KHÁC. Nhiệm vụ của Gradient
# Descent (Phần 1.5) là tìm (w, b) làm cho con số NÀY càng nhỏ càng tốt.
#
# TÓM TẮT 1 CÂU: Cost Function J(w,b) đo "độ tệ" của một đường thẳng bằng
# trung bình bình phương sai số giữa dự đoán và thực tế - J càng NHỎ, đường
# thẳng càng "khớp" với dữ liệu.

# %%
# ==============================================================================
# PHẦN 1.5 - GRADIENT DESCENT (HẠ GRADIENT) - THUẬT TOÁN TỐI ƯU
# ==============================================================================
#
# BÀI TOÁN: ta có hàm mất mát J(w, b) từ Phần 1.4. Ta MUỐN tìm (w, b) làm
# J NHỎ NHẤT có thể (gọi là "minimize J"). Gradient Descent là MỘT THUẬT
# TOÁN (không phải công thức tính trực tiếp) để làm việc này bằng cách LẶP
# ĐI LẶP LẠI, mỗi lần "nhích" (w, b) một chút theo hướng làm J giảm xuống.
#
# ẨN DỤ HÌNH ẢNH (rất kinh điển, dùng để nhớ trực giác):
# Tưởng tượng bạn đứng trên một quả đồi (địa hình chính là đồ thị của hàm
# J(w,b), càng thấp càng tốt) trong SƯƠNG MÙ DÀY ĐẶC (không nhìn thấy toàn
# cảnh, không biết đáy đồi ở đâu). Chiến lược hợp lý nhất: nhìn xuống CHÂN
# mình, xác định hướng DỐC NHẤT đi xuống, bước 1 bước theo hướng đó, rồi lặp
# lại. Đó CHÍNH XÁC là Gradient Descent: "gradient" (đạo hàm) cho biết hướng
# dốc NHẤT (nhưng là hướng đi LÊN), nên ta đi theo hướng NGƯỢC LẠI (trừ đi)
# để xuống dốc.
#
# ------------------------------------------------------------------------------
# BƯỚC TOÁN HỌC 1: TÍNH ĐẠO HÀM RIÊNG (PARTIAL DERIVATIVE) CỦA J THEO w
# ------------------------------------------------------------------------------
# Đạo hàm riêng dJ/dw trả lời câu hỏi: "nếu tăng w lên MỘT CHÚT XÍU, thì J
# thay đổi (tăng/giảm) NHANH đến mức nào?" Đây chính là "độ dốc" ở ẩn dụ trên.
#
# Ta có: J(w,b) = (1/2m) * SUM (w*x_i + b - y_i)^2
#
# Đặt u_i = w*x_i + b - y_i (để dễ nhìn công thức). Ta cần đạo hàm của u_i^2
# theo w. Đây là ĐẠO HÀM HÀM HỢP (chain rule) - quy tắc: đạo hàm của u^2 là
# 2*u nhân với đạo hàm của u (theo biến đang xét).
#
#     d/dw [u_i^2] = 2 * u_i * (d u_i / dw)
#
# Vì u_i = w*x_i + b - y_i, đạo hàm của u_i theo w là: d(u_i)/dw = x_i (vì
# b và y_i không chứa w nên đạo hàm bằng 0, chỉ có w*x_i có đạo hàm là x_i).
#
#     d/dw [u_i^2] = 2 * (w*x_i + b - y_i) * x_i = 2 * (y_hat_i - y_i) * x_i
#
# Thay vào công thức J đầy đủ (nhớ có hệ số 1/2m và dấu SUM đứng trước):
#
#     dJ/dw = (1/2m) * SUM [ 2 * (y_hat_i - y_i) * x_i ]
#           = (1/m)   * SUM [ (y_hat_i - y_i) * x_i ]      <-- số 2 đã TRIỆT
#                                                               TIÊU với mẫu
#                                                               số 2m, đúng
#                                                               như đã hứa ở
#                                                               Phần 1.4 mục 5
#
# ------------------------------------------------------------------------------
# BƯỚC TOÁN HỌC 2: TÍNH ĐẠO HÀM RIÊNG CỦA J THEO b (tương tự, đơn giản hơn)
# ------------------------------------------------------------------------------
# d(u_i)/db = 1 (vì b có hệ số 1 trong biểu thức w*x_i + b - y_i)
#
#     dJ/db = (1/2m) * SUM [ 2 * (y_hat_i - y_i) * 1 ]
#           = (1/m)   * SUM (y_hat_i - y_i)
#
# ------------------------------------------------------------------------------
# BƯỚC 3: QUY TẮC CẬP NHẬT (UPDATE RULE)
# ------------------------------------------------------------------------------
# Mỗi vòng lặp, ta cập nhật ĐỒNG THỜI cả w và b theo hướng NGƯỢC với đạo hàm
# (vì đạo hàm chỉ hướng ĐI LÊN, ta cần đi XUỐNG nên có dấu TRỪ):
#
#     w := w - alpha * (dJ/dw)
#     b := b - alpha * (dJ/db)
#
# "alpha" (ky hieu toan hoc thuong viet la chu Hy Lap alpha) goi la LEARNING
# RATE (toc do hoc) - MOT SIEU THAM SO (hyperparameter, da hoc o Phan 0.4) do
# CON NGUOI tu chon truoc, quyet dinh MOI BUOC "nhich" bao xa.
#
# SO SÁNH: LEARNING RATE QUÁ LỚN vs QUÁ NHỎ vs VỪA PHẢI
#   - alpha QUÁ NHỎ: mỗi bước nhích rất ít -> cần RẤT NHIỀU vòng lặp mới tới
#     đáy -> tốn thời gian tính toán, đôi khi "hết giờ" (hết số vòng lặp cho
#     phép) mà VẪN CHƯA tới đáy.
#   - alpha QUÁ LỚN: mỗi bước nhích quá xa, có thể NHẢY VƯỢT QUA đáy sang
#     phía bên kia, rồi lại nhảy vượt qua lần nữa theo hướng ngược lại ->
#     "dao động" (oscillate) không bao giờ tới đáy, thậm chí có thể "phân kỳ"
#     (diverge) - cost càng lúc càng TĂNG thay vì giảm.
#   - alpha VỪA PHẢI: giảm đều đặn tới đáy trong số vòng lặp hợp lý - đây
#     chính là điều ta sẽ THẤY bằng số liệu thật ngay bên dưới.
#
# ĐIỀU GÌ "VỠ" NẾU BỎ QUA HỌC PHẦN GRADIENT DESCENT? Với CHỈ 1 feature như ở
# đây, thực ra có công thức tính trực tiếp (Normal Equation, học ở Phần
# 1.10) không cần lặp. NHƯNG rất nhiều mô hình ML phức tạp hơn (Logistic
# Regression ở Phần 2, Neural Network ở Phần 7) KHÔNG CÓ công thức tính trực
# tiếp - Gradient Descent (và các biến thể của nó) là cách DUY NHẤT để huấn
# luyện chúng. Hiểu chắc Gradient Descent ở bài toán ĐƠN GIẢN này chính là
# nền tảng bắt buộc cho mọi thuật toán phức tạp hơn sau này.

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
        y_hat = w * x + b                          # du doan hien tai voi (w,b) hien tai

        # --- dung DUNG 2 cong thuc dao ham da suy ra o tren ---
        dw = np.sum((y_hat - y) * x) / m            # dJ/dw
        db = np.sum(y_hat - y) / m                  # dJ/db

        # --- cap nhat DONG THOI w va b (dung 2 bien tam de KHONG dung w MOI
        # de tinh db, vi db phai duoc tinh voi w CU) - o day vi ta da tinh
        # xong ca dw va db TRUOC khi cap nhat nen khong bi loi nay ---
        w = w - alpha * dw
        b = b - alpha * db

        cost_history.append(compute_cost(x, y, w, b))

    return w, b, cost_history

# CHẠY THẬT với dữ liệu bmi (353 bệnh nhân trong tập train)
w_final, b_final, cost_history = gradient_descent_1var(
    x_bmi_train, y_train,
    w_init=0.0, b_init=0.0,     # bat dau tu duong thang "vo nghia" y_hat=0
    alpha=0.8,                   # learning rate - da thu nghiem, hoi tu tot
    iterations=10000
)

print("Cost tai vong lap dau tien (i=0):", round(cost_history[0], 4))
print("Cost tai vong lap thu 1000  (i=999):", round(cost_history[999], 4))
print("Cost tai vong lap CUOI (i=9999):", round(cost_history[-1], 4))
print()
print(f"KET QUA CUOI: w = {w_final:.4f}, b = {b_final:.4f}")

# GIẢI THÍCH TIẾN TRÌNH HỘI TỤ VỚI SỐ LIỆU THẬT (đã chạy thử nghiệm để lấy
# đúng các mốc dưới đây - alpha=0.8, khởi tạo w=0, b=0):
#
#   vong lap 0     : w=1.9938,   b=122.9892,  cost=3506.36
#   vong lap 1     : w=3.8133,   b=147.5843,  cost=3048.61
#   vong lap 2     : w=5.5954,   b=152.5008,  cost=3026.53
#   vong lap 9     : w=17.9241,  b=153.7092,  cost=2998.67
#   vong lap 99    : w=163.4239, b=153.4561,  cost=2704.27
#   vong lap 999   : w=830.9774, b=152.2949,  cost=1958.36
#   vong lap 2999  : w=993.8539, b=152.0116,  cost=1927.08
#   vong lap 9999  : w=998.5777, b=152.0034,  cost=1927.06  <-- da HOI TU
#
# QUAN SÁT RẤT QUAN TRỌNG (một hiện tượng thật, không phải lý thuyết suông):
# Nhìn kỹ sẽ thấy "b" hội tụ RẤT NHANH (từ vòng lặp 2 đã gần 152.5, sát giá
# trị cuối 152.0034), NHƯNG "w" hội tụ CHẬM HƠN NHIỀU (vòng lặp 999 mới chỉ
# w=830, phải đến vòng ~5000-9999 mới chạm 998.58). LÝ DO: b nằm trên trục có
# "độ cong" (curvature) của hàm J lớn hơn nhiều so với w (vì feature bmi đã
# được sklearn chuẩn hoá về std rất nhỏ = 0.0476 - đã thấy ở Phần 1.0/1.1),
# khiến hướng "w" của mặt cong J RẤT THOẢI (gần phẳng) trong khi hướng "b"
# RẤT DỐC -> cùng 1 learning rate alpha, b tiến rất nhanh tới đáy còn w tiến
# rất chậm. Đây CHÍNH LÀ lý do thực tế người ta hay CHUẨN HOÁ (scale) dữ liệu
# trước khi chạy Gradient Descent - sẽ bàn kỹ hơn ở Phần 1.17.
#
# cost giảm dần qua từng vòng lặp (14855 lúc w=b=0, xuống 3506 chỉ sau ĐÚNG 1
# vòng lặp, rồi tiếp tục giảm chậm dần và "phẳng ra" ở khoảng 1927.06) -
# CHÍNH LÀ BẰNG CHỨNG bằng số cho việc thuật toán đang hoạt động ĐÚNG: mỗi
# bước đều làm J nhỏ đi (hoặc giữ nguyên khi đã tới đáy), KHÔNG BAO GIỜ tăng
# lên - nếu thấy cost TĂNG giữa chừng, đó là dấu hiệu alpha đang QUÁ LỚN.

# %%
# SO SÁNH VỚI SKLEARN: kiểm chứng code "from scratch" có đúng hay không
lr_bmi_only = LinearRegression()
lr_bmi_only.fit(x_bmi_train.reshape(-1, 1), y_train)   # sklearn yeu cau X 2 chieu,
                                                         # nen reshape (353,) -> (353,1)
print("Ket qua TU VIET (from scratch)  : w =", round(w_final,4), ", b =", round(b_final,4))
print("Ket qua thu vien sklearn        : w =", round(lr_bmi_only.coef_[0],4), ", b =", round(lr_bmi_only.intercept_,4))

# GIẢI THÍCH: w_scratch=998.5777 và w_sklearn=998.5777 (khớp đến 4 chữ số
# thập phân!), b_scratch=152.0034 và b_sklearn=152.0034 -> KHỚP HOÀN TOÀN.
# Đây LÀ BẰNG CHỨNG code Gradient Descent tự viết ở trên đã ĐÚNG - sklearn
# không dùng Gradient Descent cho Linear Regression (nó dùng một phương pháp
# đại số tuyến tính trực tiếp, gần giống Normal Equation ở Phần 1.10), nhưng
# vì cả 2 đều đang giải ĐÚNG MỘT bài toán tối ưu (minimize cùng 1 hàm J), nên
# nếu Gradient Descent đã HỘI TỤ THẬT SỰ (không dừng giữa chừng), kết quả
# PHẢI khớp với sklearn - đây là cách ta "tự chấm điểm" cho code tự viết.

# %%
# ==============================================================================
# PHẦN 1.6 - TRỰC QUAN HOÁ: ĐƯỜNG CONG HỘI TỤ (CONVERGENCE PLOT)
# ==============================================================================
#
# TẠI SAO CẦN VẼ CONVERGENCE PLOT? Nhìn dãy số cost_history khô khan rất khó
# đánh giá "thuật toán có ổn không". Vẽ đồ thị cost theo từng vòng lặp cho
# phép NHÌN BẰNG MẮT ngay lập tức: cost có giảm đều đặn không, có bị "tăng
# vọt" (dấu hiệu alpha quá lớn) không, và đã "phẳng ra" (hội tụ) hay chưa.

plt.figure(figsize=(10, 5))
plt.plot(range(len(cost_history)), cost_history, color='steelblue', linewidth=2)
plt.xlabel('Vong lap (iteration)')
plt.ylabel('Cost J(w, b)')
plt.title('Duong cong hoi tu cua Gradient Descent (Simple Linear Regression - bmi)')
plt.grid(True, alpha=0.3)
plt.show()

# GIẢI THÍCH HÌNH DẠNG ĐỒ THỊ: đường cong sẽ giảm RẤT NHANH và DỐC ở những
# vòng lặp đầu tiên (từ 14855 gần như rơi thẳng xuống dưới 3506 chỉ sau 1
# vòng lặp), sau đó độ dốc thoải dần và đường cong gần như NẰM NGANG (phẳng)
# từ khoảng vòng lặp 3000 trở đi quanh giá trị ~1927. "Nằm ngang" chính là
# dấu hiệu HỘI TỤ (converged) - tiếp tục lặp thêm cũng không cải thiện được
# gì nhiều nữa, vì đã rất gần điểm cực tiểu thật sự của hàm J.

# %%
# TRỰC QUAN HOÁ: ĐƯỜNG HỒI QUY TRÊN DỮ LIỆU THẬT
plt.figure(figsize=(10, 6))
plt.scatter(x_bmi_train, y_train, alpha=0.5, color='cornflowerblue', label='Du lieu that (353 benh nhan, tap train)')

# ve duong thang y_hat = w*x + b tren cung do thi
x_line = np.linspace(x_bmi_train.min(), x_bmi_train.max(), 100)
y_line = w_final * x_line + b_final
plt.plot(x_line, y_line, color='crimson', linewidth=2.5,
         label=f'Duong hoi quy: y_hat = {w_final:.1f}*x + {b_final:.1f}')

plt.xlabel('BMI (da chuan hoa boi sklearn)')
plt.ylabel('Muc do tien trien benh (target)')
plt.title('Simple Linear Regression: BMI vs Muc do tien trien benh tieu duong (du lieu THAT)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# GIẢI THÍCH: mỗi chấm xanh là 1 bệnh nhân THẬT trong tập train (bmi của họ
# trên trục x, mức độ bệnh thật trên trục y). Đường đỏ chính là "kết luận"
# mà Gradient Descent tìm ra: nó KHÔNG đi qua CHÍNH XÁC từng điểm (vì dữ liệu
# y tế thực tế luôn có nhiễu/biến động cá nhân), nhưng nó đi qua GIỮA đám
# mây điểm theo hướng "khớp" nhất có thể theo tiêu chí MSE đã học ở Phần 1.4.
# Độ dốc DƯƠNG của đường thẳng (w=998.98 > 0) khẳng định lại điều đã thấy ở
# hệ số tương quan Phần 1.1 (r=0.5865 > 0): bmi càng cao, mô hình càng dự
# đoán mức độ bệnh cao hơn.


# %%
# ==============================================================================
# PHẦN 1.7 - THEO DẤU MỘT VÍ DỤ CỤ THỂ (CONCRETE TRACE) - 1 BỆNH NHÂN THẬT
# ==============================================================================
#
# Để hiểu THẬT SÂU công thức y_hat = w*x + b không chỉ là ký hiệu trừu tượng,
# hãy theo dấu quá trình dự đoán cho ĐÚNG 1 bệnh nhân thật, từng bước một.

patient_x = x_bmi_train[0]     # gia tri bmi (da chuan hoa) cua benh nhan dau tien trong tap train
patient_y_true = y_train[0]    # gia tri target THAT cua benh nhan do

print(f"Benh nhan train[0]:")
print(f"  bmi (da chuan hoa) = {patient_x}")
print(f"  target THAT        = {patient_y_true}")
print()

prediction = w_final * patient_x + b_final
print(f"Ap dung cong thuc: y_hat = w * x + b")
print(f"                        = {w_final:.4f} * {patient_x:.6f} + {b_final:.4f}")
print(f"                        = {w_final*patient_x:.4f} + {b_final:.4f}")
print(f"                        = {prediction:.4f}")
print()
error = prediction - patient_y_true
print(f"Sai so (prediction - true) = {prediction:.4f} - {patient_y_true} = {error:.4f}")

# GIẢI THÍCH BẰNG SỐ THẬT (đã tính chính xác ở trên, không làm tròn tuỳ
# tiện):
#   x (bmi chuan hoa) = 0.012117
#   y that            = 144.0
#   y_hat = 998.5777 * 0.012117 + 152.0034
#         = 12.0996 + 152.0034
#         = 164.1030
#   sai so = 164.1030 - 144.0 = +20.10
#
# Ý NGHĨA: với bệnh nhân THẬT này, mô hình dự đoán mức độ bệnh là 164.10,
# trong khi con số thật là 144.0 -> mô hình dự đoán CAO HƠN thực tế 20.10
# đơn vị. Đây là 1 sai số ĐƠN LẺ - hoàn toàn bình thường, vì mô hình CHỈ
# DÙNG 1 feature (bmi) trong khi mức độ bệnh thật còn phụ thuộc vào 9 chỉ số
# y tế KHÁC nữa (tuổi, huyết áp, các chỉ số huyết thanh...) mà mô hình đơn
# biến này CHƯA XÉT ĐẾN - đây CHÍNH LÀ động lực để mở rộng sang Hồi quy
# tuyến tính ĐA BIẾN ở Phần 1.9 ngay sau đây, dùng ĐỦ CẢ 10 feature để dự
# đoán chính xác hơn.
#
# TÓM TẮT 1 CÂU PHẦN 1.3 - 1.8: Simple Linear Regression tìm 1 đường thẳng
# (w, b) tối thiểu hoá MSE bằng Gradient Descent (lặp dần theo hướng ngược
# đạo hàm), và kết quả khớp CHÍNH XÁC với công thức sklearn khi đã hội tụ
# thật sự - nhưng dùng CHỈ 1 feature vẫn còn để sót nhiều thông tin.

# %%
# ==============================================================================
# PHẦN 1.8 - LY THUYET: HOI QUY TUYEN TINH DA BIEN (MULTIPLE LINEAR REGRESSION)
# ==============================================================================
#
# Bây giờ dùng ĐỦ CẢ 10 feature thay vì chỉ 1. Công thức mở rộng tự nhiên từ
# Phần 1.3:
#
#     y_hat = w1*x1 + w2*x2 + w3*x3 + ... + w10*x10 + b
#
# Với 10 features, viết ra 10 số hạng w_i*x_i rất DÀI DÒNG. Ta dùng KÝ HIỆU
# VECTOR để viết gọn (đây chính là lý do "vector hoá" - vectorization - lại
# quan trọng, và cũng là lý do numpy tồn tại):
#
#     y_hat = W^T . X + b    (hoac viet: y_hat = X . W + b, tuy quy uoc)
#
# Trong do:
#   X = [x1, x2, ..., x10]   (vector 10 chieu, cac feature cua 1 benh nhan)
#   W = [w1, w2, ..., w10]   (vector 10 chieu, CAC THAM SO can hoc - moi
#                              feature co 1 trong so RIENG cua no)
#   W^T . X (hoac X . W)     : TICH VO HUONG (dot product) giua 2 vector,
#                              chinh la w1*x1 + w2*x2 + ... + w10*x10
#
# Ý NGHĨA CỦA TỪNG w_i: w_i cho biết "nếu x_i tăng 1 đơn vị, VÀ MỌI feature
# khác GIỮ NGUYÊN, thì y_hat thay đổi bao nhiêu". Ví dụ nếu w_bmi = 542 (sẽ
# thấy con số thật ngay dưới đây), nghĩa là ứng với mỗi đơn vị bmi (đã chuẩn
# hoá) tăng thêm, mức độ bệnh dự đoán tăng thêm khoảng 542 đơn vị, NẾU các
# chỉ số khác không đổi.
#
# HÀM MẤT MÁT (mở rộng tự nhiên từ Phần 1.4, chỉ thay y_hat_i bằng công thức
# đa biến, công thức TỔNG QUÁT giữ nguyên):
#
#     J(W, b) = (1/2m) * SUM[i=1 den m] (y_hat_i - y_i)^2
#
# ĐẠO HÀM (dạng vector hoá - đây là kết quả TỔNG QUÁT của chính đạo hàm đã
# suy ra ở Phần 1.5, chỉ khác là bây giờ có 10 đạo hàm riêng dJ/dw1...dJ/dw10
# thay vì chỉ 1, và numpy cho phép tính CẢ 10 đạo hàm đó CÙNG LÚC bằng phép
# nhân ma trận, KHÔNG cần viết 10 dòng code riêng biệt):
#
#     dJ/dW = (1/m) * X^T . (y_hat - y)      (dJ/dW la 1 vector 10 chieu)
#     dJ/db = (1/m) * SUM (y_hat - y)         (dJ/db van la 1 so thuc)
#
# Quy tắc cập nhật (giống hệt ý tưởng Phần 1.5, áp dụng cho CẢ VECTOR W):
#
#     W := W - alpha * (dJ/dW)
#     b := b - alpha * (dJ/db)

def compute_cost_multi(X, y, w, b):
    """
    Ham mat mat MSE cho Hoi quy tuyen tinh DA BIEN.
    X : ma tran (m, n) - m sample, n feature
    w : vector (n,) - n trong so, moi feature 1 trong so
    b : 1 so thuc
    """
    m = X.shape[0]
    y_hat = X.dot(w) + b     # X.dot(w) tinh CUNG LUC ca 353 gia tri du doan,
                              # moi gia tri la 1 tich vo huong giua 1 hang cua
                              # X (10 feature cua 1 benh nhan) voi vector w
    cost = np.sum((y_hat - y) ** 2) / (2 * m)
    return cost

def gradient_descent_multi(X, y, w_init, b_init, alpha, iterations):
    """Gradient Descent cho Hoi quy tuyen tinh DA BIEN (vector hoa)."""
    m = X.shape[0]
    w, b = w_init.copy(), b_init
    cost_history = []

    for i in range(iterations):
        y_hat = X.dot(w) + b
        error = y_hat - y                      # vector (m,), sai so cua TUNG benh nhan

        dw = X.T.dot(error) / m                 # X.T co shape (10, m); nhan voi error
                                                  # (m,) cho ra vector (10,) - CHINH LA
                                                  # ca 10 dao ham rieng dJ/dw1...dJ/dw10
                                                  # tinh CUNG LUC, khong can 10 dong code
        db = np.sum(error) / m

        w = w - alpha * dw
        b = b - alpha * db

        cost_history.append(compute_cost_multi(X, y, w, b))

    return w, b, cost_history

# CHẠY THẬT với đủ 10 feature
n_features = X_train.shape[1]
w_init_multi = np.zeros(n_features)     # khoi tao TAT CA 10 trong so bang 0
w_multi, b_multi, cost_history_multi = gradient_descent_multi(
    X_train, y_train, w_init_multi, 0.0, alpha=0.5, iterations=30000
)

for i in [999, 2999, 9999, 19999, 29999]:
    print(f"vong lap {i}: cost = {cost_history_multi[i]:.4f}")
print()
print("W cuoi cung (10 trong so, theo thu tu", feature_names, "):")
print(w_multi.round(3))
print("b cuoi cung:", round(b_multi, 3))

# GIẢI THÍCH SỐ LIỆU THẬT: cost giảm từ ~5981 (vòng lặp đầu) xuống dần
# 1511.38 (vòng 999) -> 1450.43 (vòng 2999) -> 1446.98 (vòng 9999) -> 1444.58
# (vòng 19999) -> 1442.63 (vòng 29999). Cost VẪN đang giảm (dù rất chậm) sau
# tới 30000 vòng lặp - khác hẳn với trường hợp 1 feature ở Phần 1.5 (đã hội
# tụ hoàn toàn chỉ sau ~5000-9999 vòng). PHẦN 1.11 ngay sau đây sẽ giải thích
# CHÍNH XÁC tại sao đa biến hội tụ chậm hơn nhiều, bằng một hiện tượng THẬT
# đã thấy thoáng qua ở heatmap Phần 1.1.


# %%
# ==============================================================================
# PHẦN 1.9 - NORMAL EQUATION (NGHIỆM DẠNG ĐÓNG - CLOSED-FORM SOLUTION)
# ==============================================================================
#
# Gradient Descent là phương pháp LẶP (từ từ tiến gần đáy). Nhưng với RIÊNG
# bài toán Linear Regression (không phải mọi mô hình ML), có một công thức
# TOÁN HỌC tính THẲNG ra (w, b) tối ưu chỉ bằng 1 PHÉP TÍNH duy nhất, không
# cần lặp - gọi là Normal Equation (Phương trình chuẩn tắc).
#
# Ý TƯỞNG: hàm J(W) (gộp cả b vào W bằng mẹo thêm 1 cột toàn số 1 vào X, giải
# thích ngay dưới) là một hàm PARABOL (lồi - convex) nhiều chiều - nó chỉ có
# DUY NHẤT 1 điểm cực tiểu (không có "cực tiểu giả"/local minimum nào khác
# để bị mắc kẹt). Tại điểm cực tiểu đó, đạo hàm PHẢI BẰNG 0 (giống hệt tìm
# cực trị hàm 1 biến trong Toán phổ thông: y'=0). Ta GIẢI TRỰC TIẾP phương
# trình "đạo hàm = 0" thay vì dò dần bằng Gradient Descent.
#
# MẸO GỘP b VÀO W: thêm 1 cột toàn giá trị 1 vào ma trận X (gọi cột đó là
# x0=1 cho MỌI bệnh nhân), khi đó w0*x0 = w0*1 = w0 CHÍNH LÀ vai trò của b -
# nhờ vậy công thức chỉ còn y_hat = X_b . Theta (voi Theta = [b, w1,...,w10]
# la 1 vector duy nhat gop ca b lan 10 trong so w, va X_b la X co them cot 1
# o dau).
#
# Đặt đạo hàm dJ/dTheta = 0 và giải (bỏ qua các bước biến đổi đại số ma trận
# chi tiết - đây là kết quả cuối, có thể tra cứu lại trong CS229/ISLR nếu
# muốn xem đầy đủ phần chứng minh):
#
#     Theta = (X_b^T . X_b)^(-1) . X_b^T . y
#
# Trong do (.)^(-1) la NGHICH DAO CUA MA TRAN (matrix inverse) - tuong tu
# nhu phep chia trong so hoc thong thuong, nhung danh cho ma tran.

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]   # them cot toan so 1
                                                              # vao DAU X_train
                                                              # (np.c_ ghep cot)
theta_normal_eq = np.linalg.inv(X_train_bias.T.dot(X_train_bias)).dot(X_train_bias.T).dot(y_train)

print("X_train_bias.shape (co them 1 cot so 1):", X_train_bias.shape)
print()
print("Theta tu Normal Equation (b, w1, w2, ..., w10):")
print(theta_normal_eq.round(3))

# GIẢI THÍCH CON SỐ THẬT:
# Theta = [151.346, 37.904, -241.964, 542.429, 347.704, -931.489, 518.062,
#          163.42, 275.318, 736.199, 48.671]
# Phan tu DAU TIEN (151.346) chinh la b (vi cot dau tien cua X_train_bias la
# cot so 1, tuong ung voi vi tri "b" trong Theta). 10 phan tu con lai la
# w1...w10 tuong ung 10 feature theo dung thu tu ['age','sex','bmi','bp',
# 's1','s2','s3','s4','s5','s6'].
#
# SO SÁNH: GRADIENT DESCENT (30000 vòng) vs NORMAL EQUATION
#   b       : GD=151.313   vs  NE=151.346   -> RAT GAN nhau
#   w_bmi   : GD=554.718   vs  NE=542.429   -> gan nhau
#   w_s1    : GD=-296.61   vs  NE=-931.489  -> KHAC NHAU RAT NHIEU!
#   w_s2    : GD=20.561    vs  NE=518.062   -> KHAC NHAU RAT NHIEU!
# Một vài trọng số (đặc biệt w_s1, w_s2) khác nhau RẤT LỚN dù cả 2 phương
# pháp đều đang cố giải CÙNG MỘT bài toán tối ưu! Đây KHÔNG PHẢI lỗi code -
# Phần 1.10 ngay sau đây sẽ giải thích chính xác hiện tượng này.
#
# TÓM TẮT 1 CÂU: Normal Equation cho ra (w,b) tối ưu bằng ĐÚNG 1 phép tính
# đại số tuyến tính (không cần lặp, không cần chọn alpha) - nhưng chỉ áp
# dụng được cho riêng Linear Regression, không dùng được cho Logistic
# Regression hay Neural Network ở các phần sau.

# %%
# ==============================================================================
# PHẦN 1.10 - TAI SAO GD VA NORMAL EQUATION CHO TRONG SO KHAC NHAU?
#              (BAI HOC THAT VE DA CONG TUYEN - MULTICOLLINEARITY)
# ==============================================================================
#
# Đây là một hiện tượng THẬT, quan sát được trên chính dữ liệu 442 bệnh nhân
# này - không phải dàn dựng. Hãy quay lại observation đã ghi chú ở Phần 1.1:
# heatmap cho thấy 2 cột "s1" và "s2" có màu đỏ RẤT ĐẬM. Kiểm tra lại bằng số:

print("He so tuong quan giua s1, s2, s3, s4 (cac chi so huyet thanh):")
print(df[['s1','s2','s3','s4']].corr().round(2))

# GIẢI THÍCH SỐ LIỆU THẬT:
#   corr(s1, s2) = 0.89   -> TƯƠNG QUAN RẤT CAO (gần 1)
#   corr(s3, s4) = -0.74  -> tương quan khá cao (âm)
#
# HIỆN TƯỢNG NÀY GỌI LÀ MULTICOLLINEARITY (ĐA CỘNG TUYẾN): khi 2 (hoặc
# nhiều) feature gần như "mang cùng 1 thông tin" (feature này gần như đoán
# được từ feature kia), mô hình Linear Regression RẤT KHÓ tách bạch "hiệu
# ứng thật sự" thuộc về feature nào.
#
# ẨN DỤ ĐỂ HIỂU TẠI SAO: tưởng tượng bạn thuê 2 người X và Y LUÔN LUÔN làm
# việc CÙNG NHAU, kết quả công việc rất tốt. Bạn KHÔNG THỂ biết chính xác
# "công lao thuộc về X bao nhiêu %, Y bao nhiêu %" - có thể nói X đóng góp
# 90% Y đóng góp 10%, hoặc X 10% Y 90%, hoặc 50-50 - TẤT CẢ đều cho ra CÙNG
# MỘT kết quả công việc cuối cùng, vì X và Y luôn "bù trừ" cho nhau. Đó CHÍNH
# XÁC là điều xảy ra với s1 và s2: có RẤT NHIỀU cặp (w_s1, w_s2) khác nhau
# đều cho ra gần như CÙNG MỘT w_s1*s1 + w_s2*s2 (vì s1 ~ s2 khi 2 biến tương
# quan cao) -> hàm J(W) có một "THUNG LŨNG DÀI VÀ PHẲNG" theo hướng (w_s1,
# w_s2) thay vì 1 đáy rõ ràng -> Gradient Descent "lang thang" rất lâu trong
# thung lũng phẳng đó mà cost giảm RẤT CHẬM (đúng như đã thấy ở Phần 1.8: cost
# vẫn giảm sau 30000 vòng lặp) -> vì DỪNG GIỮA CHỪNG (chưa đi hết thung
# lũng), GD cho ra 1 điểm BẤT KỲ trong thung lũng đó, khác với điểm mà Normal
# Equation tính THẲNG RA (điểm chính giữa đáy thật sự).
#
# CÂU HỎI QUAN TRỌNG NHẤT: vậy kết quả DỰ ĐOÁN có bị ảnh hưởng không? Kiểm
# tra bằng cách so sánh hiệu suất trên tập TEST (89 bệnh nhân CHƯA TỪNG thấy):

y_pred_test_gd = X_test.dot(w_multi) + b_multi
y_pred_test_ne = X_test.dot(theta_normal_eq[1:]) + theta_normal_eq[0]

r2_gd = r2_score(y_test, y_pred_test_gd)
r2_ne = r2_score(y_test, y_pred_test_ne)
mse_gd = mean_squared_error(y_test, y_pred_test_gd)
mse_ne = mean_squared_error(y_test, y_pred_test_ne)

print(f"R2 tren test set  - Gradient Descent : {r2_gd:.4f}")
print(f"R2 tren test set  - Normal Equation   : {r2_ne:.4f}")
print(f"MSE tren test set - Gradient Descent  : {mse_gd:.3f}")
print(f"MSE tren test set - Normal Equation   : {mse_ne:.3f}")

# GIẢI THÍCH KẾT QUẢ THẬT (đây là điểm MẤU CHỐT của cả bài học này):
#   R2 (GD)  = 0.4552      R2 (NE)  = 0.4526    -> GAN NHU BANG NHAU
#   MSE (GD) = 2886.303    MSE (NE) = 2900.194  -> GAN NHU BANG NHAU
#
# Dù 2 bộ trọng số (w_s1, w_s2...) rất KHÁC NHAU về mặt CON SỐ, hiệu suất DỰ
# ĐOÁN trên dữ liệu MỚI (test set) lại GẦN NHƯ GIỐNG HỆT NHAU! Đây chính là
# bài học cốt lõi của multicollinearity, RẤT quan trọng trong công việc Data
# Science thực tế:
#
#   -> Multicollinearity làm cho VIỆC DIỄN GIẢI từng hệ số w_i riêng lẻ trở
#      nên KHÔNG ĐÁNG TIN CẬY (không thể nói chắc "s1 quan trọng hơn s2 bao
#      nhiêu", vì con số w_s1, w_s2 phụ thuộc vào cách thuật toán "tình cờ"
#      dừng lại ở đâu trong thung lũng phẳng).
#   -> NHƯNG multicollinearity KHÔNG nhất thiết làm hại KHẢ NĂNG DỰ ĐOÁN
#      tổng thể của mô hình - vì y_hat = w1*x1+w2*x2+... vẫn cho ra kết quả
#      tương tự dù (w1, w2) là cặp giá trị nào trong "thung lũng" đó.
#
# ĐÂY LÀ LÝ DO thực tế các Data Scientist khi làm việc với dữ liệu có nhiều
# feature tương quan cao, sẽ CẨN TRỌNG khi diễn giải "feature nào quan
# trọng nhất" chỉ dựa vào độ lớn hệ số hồi quy, và thường cân nhắc loại bớt
# feature dư thừa (sẽ bàn ở Phần 5 - Feature Engineering) hoặc dùng
# Regularization (Phần 1.15-1.16 ngay bên dưới, cũng giải quyết được phần
# nào vấn đề này).


# %%
# ==============================================================================
# PHẦN 1.11 - SKLEARN LINEARREGRESSION (DA BIEN) - DOI CHIEU LAN CUOI
# ==============================================================================

lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)

print("sklearn coef_ (w1...w10):")
print(lr_multi.coef_.round(3))
print("sklearn intercept_ (b):", round(lr_multi.intercept_, 3))
print()
print("So sanh voi Normal Equation tu viet:")
print("Normal Eq w:", theta_normal_eq[1:].round(3))
print("Normal Eq b:", round(theta_normal_eq[0], 3))

# GIẢI THÍCH: sklearn.coef_ = [37.904, -241.964, 542.429, 347.704, -931.489,
# 518.062, 163.42, 275.318, 736.199, 48.671], intercept_ = 151.346 -> KHỚP
# CHÍNH XÁC (đến 3 chữ số thập phân) với Theta tự tính bằng Normal Equation ở
# Phần 1.9! Điều này xác nhận: sklearn.linear_model.LinearRegression, dù bên
# trong dùng thuật toán tối ưu hơn (phân rã SVD - Singular Value
# Decomposition - để ổn định số học tốt hơn khi ma trận gần suy biến do
# multicollinearity), về mặt TOÁN HỌC đang giải ĐÚNG bài toán mà Normal
# Equation giải - và code tự viết ở Phần 1.9 đã cho ra kết quả ĐÚNG.
#
# TÓM TẮT 1 CÂU PHẦN 1.8-1.11: mở rộng sang đa biến chỉ thay đổi ký hiệu
# (vector thay vì số), Normal Equation cho nghiệm CHÍNH XÁC ngay lập tức
# (khớp sklearn), còn Gradient Descent tuy hội tụ chậm hơn khi feature tương
# quan cao (multicollinearity) nhưng vẫn cho hiệu suất dự đoán tương đương.

# %%
# ==============================================================================
# PHẦN 1.12 - CAC CHI SO DANH GIA MO HINH HOI QUY: MSE, RMSE, MAE, R2
# ==============================================================================
#
# Từ đây trở đi, ta dùng mô hình sklearn.LinearRegression (đa biến, đã khớp
# hoàn hảo với Normal Equation) để dự đoán trên TẬP TEST (89 bệnh nhân CHƯA
# TỪNG được dùng để học) - đây MỚI LÀ phép đánh giá công bằng, đúng tinh thần
# đã học ở Phần 1.2 và Phần 0.3.

y_pred_test = lr_multi.predict(X_test)

print("5 gia tri target THAT (test):", y_test[:5])
print("5 gia tri DU DOAN (test)    :", y_pred_test[:5].round(2))
print("5 sai so tuyet doi          :", np.abs(y_test[:5] - y_pred_test[:5]).round(2))

# ------------------------------------------------------------------------------
# (A) MSE - MEAN SQUARED ERROR (đã học công thức ở Phần 1.4, giờ áp dụng để
#     ĐÁNH GIÁ thay vì để HUẤN LUYỆN - CÙNG 1 công thức, 2 mục đích khác nhau)
# ------------------------------------------------------------------------------
def my_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)     # luu y: KHONG chia 2 nhu Phan
                                                  # 1.4 (khong can nua vi day
                                                  # la de DANH GIA, khong can
                                                  # dao ham) - day la quy uoc
                                                  # CHUAN cua metric MSE

mse_manual = my_mse(y_test, y_pred_test)
mse_sklearn = mean_squared_error(y_test, y_pred_test)
print(f"\nMSE tu viet: {mse_manual:.3f}  |  MSE sklearn: {mse_sklearn:.3f}  -> KHOP")

# Ý NGHĨA: MSE = 2900.194. Đơn vị của MSE là ĐƠN VỊ CỦA y BÌNH PHƯƠNG (vì có
# phép bình phương trong công thức) -> con số 2900 tự nó RẤT KHÓ diễn giải
# trực quan (2900 "đơn vị bệnh bình phương" nghĩa là gì?). Đây chính là lý do
# cần thêm RMSE ngay sau đây.
#
# ------------------------------------------------------------------------------
# (B) RMSE - ROOT MEAN SQUARED ERROR (căn bậc 2 của MSE)
# ------------------------------------------------------------------------------
def my_rmse(y_true, y_pred):
    return np.sqrt(my_mse(y_true, y_pred))     # chi don gian lay CAN BAC 2 cua MSE

rmse_manual = my_rmse(y_test, y_pred_test)
print(f"RMSE tu viet: {rmse_manual:.3f}  |  RMSE sklearn (via sqrt): {np.sqrt(mse_sklearn):.3f}")

# Ý NGHĨA: RMSE = sqrt(2900.194) = 53.853. Lấy căn bậc 2 đưa đơn vị TRỞ LẠI
# giống ĐƠN VỊ GỐC của y (không còn bị bình phương nữa) -> CÓ THỂ DIỄN GIẢI
# TRỰC QUAN: "trung bình, dự đoán của mô hình lệch khoảng 53.85 đơn vị so
# với giá trị thật" - dễ hiểu hơn hẳn con số MSE=2900.194 trần trụi.
#
# ------------------------------------------------------------------------------
# (C) MAE - MEAN ABSOLUTE ERROR (sai số tuyệt đối trung bình)
# ------------------------------------------------------------------------------
def my_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))    # dung GIA TRI TUYET DOI thay vi binh phuong

mae_manual = my_mae(y_test, y_pred_test)
mae_sklearn = mean_absolute_error(y_test, y_pred_test)
print(f"MAE tu viet: {mae_manual:.3f}  |  MAE sklearn: {mae_sklearn:.3f}  -> KHOP")

# Ý NGHĨA: MAE = 42.794. CŨNG cùng đơn vị với y (không bị bình phương), NHƯNG
# khác RMSE ở chỗ: MAE coi MỌI sai số "nặng" như nhau theo tỉ lệ thuận, còn
# RMSE (vì có bình phương rồi mới khai căn) PHẠT NẶNG HƠN các sai số LỚN bất
# thường (outlier).
#
# SO SÁNH RMSE vs MAE (BẢNG SO SÁNH, khi nào dùng cái nào):
#   RMSE (53.853) > MAE (42.794) LUÔN LUÔN đúng về mặt toán học (có thể
#   chứng minh bằng bất đẳng thức, không cần nhớ chứng minh, chỉ cần nhớ
#   TÍNH CHẤT: RMSE >= MAE luôn luôn). Khoảng CÁCH giữa RMSE và MAE (ở đây là
#   53.853 - 42.794 = 11.06) càng LỚN, nghĩa là trong tập dữ liệu có MỘT SỐ
#   dự đoán sai số RẤT LỚN (outlier) đang "kéo" RMSE lên cao hơn nhiều so với
#   MAE (vì RMSE phạt nặng các sai số lớn, còn MAE thì không).
#     - Muốn mô hình "quan tâm nhiều" tới việc tránh sai số LỚN (vd: trong y
#       tế, một dự đoán sai lệch RẤT NHIỀU có thể nguy hiểm) -> quan tâm RMSE
#       nhiều hơn.
#     - Muốn một chỉ số ÍT BỊ ẢNH HƯỞNG bởi vài điểm dữ liệu bất thường
#       (outlier) -> MAE đáng tin cậy hơn.
#
# ------------------------------------------------------------------------------
# (D) R² (R-SQUARED / HỆ SỐ XÁC ĐỊNH) - chỉ số QUAN TRỌNG NHẤT để "cảm nhận"
#     mô hình tốt tới đâu, vì nó có THANG ĐO CHUẨN (không phụ thuộc đơn vị y)
# ------------------------------------------------------------------------------
# CÔNG THỨC:
#     R^2 = 1 - (SS_res / SS_tot)
#
#   SS_res (Sum of Squares Residual) = SUM (y_i - y_hat_i)^2
#       -> tổng bình phương sai số CỦA MÔ HÌNH ta đang đánh giá
#   SS_tot (Sum of Squares Total)    = SUM (y_i - y_bar)^2   (y_bar = trung
#       binh cua y_test)
#       -> tổng bình phương sai số của một mô hình "NGÂY THƠ" nhất có thể:
#          MÔ HÌNH LUÔN DỰ ĐOÁN BẰNG TRUNG BÌNH, bất kể feature đầu vào là gì
#
# Ý TƯỞNG CỐT LÕI: R^2 SO SÁNH mô hình của ta với một "đường tham chiếu"
# (baseline) NGÂY THƠ nhất - mô hình chỉ đoán trung bình mọi lúc, hoàn toàn
# KHÔNG dùng thông tin feature gì cả.
#   R^2 = 1   -> mô hình dự đoán HOÀN HẢO (SS_res = 0)
#   R^2 = 0   -> mô hình TỆ NGANG với việc chỉ đoán trung bình (SS_res=SS_tot)
#   R^2 < 0   -> mô hình còn TỆ HƠN việc chỉ đoán trung bình (có thể xảy ra
#                nếu mô hình bị huấn luyện/chọn feature rất kém)

y_bar_test = y_test.mean()
ss_res = np.sum((y_test - y_pred_test) ** 2)
ss_tot = np.sum((y_test - y_bar_test) ** 2)
r2_manual = 1 - ss_res / ss_tot
r2_sklearn = r2_score(y_test, y_pred_test)

print(f"\ny_bar (trung binh y_test) = {y_bar_test:.3f}")
print(f"SS_res = {ss_res:.2f}")
print(f"SS_tot = {ss_tot:.2f}")
print(f"R2 tu viet: {r2_manual:.4f}  |  R2 sklearn: {r2_sklearn:.4f}  -> KHOP")

# GIẢI THÍCH CON SỐ THẬT: y_bar=145.775, SS_res=258117.23, SS_tot=471535.51
#   R^2 = 1 - 258117.23/471535.51 = 1 - 0.5474 = 0.4526
#
# DIỄN GIẢI Ý NGHĨA THỰC TẾ: R^2=0.4526 nghĩa là mô hình (dùng 10 chỉ số y
# tế) giải thích được khoảng 45.26% sự BIẾN ĐỘNG (variance) của mức độ tiến
# triển bệnh giữa các bệnh nhân - 54.74% biến động còn lại đến từ các yếu tố
# KHÔNG có trong 10 feature này (vd: chế độ ăn, di truyền, vận động...) hoặc
# nhiễu ngẫu nhiên tự nhiên trong dữ liệu y tế. R^2=0.45 không phải "hoàn hảo"
# nhưng CŨNG KHÔNG TỆ cho bài toán y tế thực tế (dữ liệu y sinh học thường có
# rất nhiều biến động khó đoán) - và quan trọng hơn, ta sẽ dùng chính con số
# 0.4526 này làm ĐIỂM CHUẨN để so sánh khi thử các kỹ thuật cải thiện ở các
# phần sau (Regularization, và cả các thuật toán mạnh hơn Linear Regression
# ở Phần 2-3).
#
# ------------------------------------------------------------------------------
# BẢNG TỔNG KẾT 4 CHỈ SỐ (tra cứu nhanh khi cần chọn chỉ số phù hợp)
# ------------------------------------------------------------------------------
#  Chi so | Cong thuc              | Don vi        | Khi nao dung
#  -------|-------------------------|---------------|---------------------------
#  MSE    | trung binh (loi)^2      | (don vi y)^2  | Dung de HUAN LUYEN (co the
#         |                         |               | lay dao ham de toi uu)
#  RMSE   | can(MSE)                 | don vi y      | Bao cao ket qua de nguoi
#         |                         |               | khac hieu, phat sai so lon
#  MAE    | trung binh |loi|         | don vi y      | It bi anh huong boi
#         |                         |               | outlier hon RMSE
#  R2     | 1 - SS_res/SS_tot        | khong don vi  | So sanh muc do "tot" cua
#         |                         | (0 den 1)     | mo hinh mot cach CHUAN HOA,
#         |                         |               | de so sanh GIUA CAC BAI
#         |                         |               | TOAN khac nhau
#
# TÓM TẮT 1 CÂU: MSE dùng để HUẤN LUYỆN (đạo hàm được), RMSE/MAE dùng để BÁO
# CÁO sai số theo đúng đơn vị gốc (khác nhau ở mức độ phạt outlier), còn R2
# cho biết mô hình tốt hơn "đoán mò trung bình" bao nhiêu phần trăm.

# %%
# ==============================================================================
# PHẦN 1.13 - OVERFITTING & UNDERFITTING: MINH HOA BANG POLYNOMIAL REGRESSION
# ==============================================================================
#
# Ở Phần 0.4 ta đã ĐỊNH NGHĨA overfitting/underfitting bằng lời. Giờ ta sẽ
# THẤY hiện tượng đó bằng con số THẬT, dùng lại feature "bmi" (đơn biến, dễ
# vẽ đồ thị) nhưng thay vì chỉ khớp 1 ĐƯỜNG THẲNG, ta khớp các ĐA THỨC
# (polynomial) bậc càng ngày càng cao, để "ép" mô hình ngày càng phức tạp.
#
# Ý TƯỞNG POLYNOMIAL REGRESSION: thay vì y_hat = w*x + b (bậc 1, đường
# thẳng), ta thêm các số hạng x^2, x^3, x^4... làm feature MỚI:
#
#     y_hat = w1*x + w2*x^2 + w3*x^3 + ... + wk*x^k + b     (bac k)
#
# LƯU Ý QUAN TRỌNG: đây VẪN LÀ Linear Regression (mô hình vẫn TUYẾN TÍNH với
# CÁC THAM SỐ w1...wk - không tuyến tính với x thôi) - PolynomialFeatures chỉ
# đơn giản TẠO RA các cột x^2, x^3... từ cột x gốc, rồi đưa vào ĐÚNG công
# thức LinearRegression đã học từ đầu. Đây KHÔNG PHẢI dữ liệu giả - x^2, x^3
# vẫn được tính TỪ giá trị bmi THẬT của từng bệnh nhân, không có gì bịa ra.

degrees_to_try = [1, 2, 3, 5, 8, 12]
train_mse_list = []
test_mse_list = []

for degree in degrees_to_try:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp_train = poly.fit_transform(x_bmi_train.reshape(-1, 1))   # tao cot x, x^2,...,x^degree
    Xp_test = poly.transform(x_bmi_test.reshape(-1, 1))          # ap dung CUNG cong thuc do
                                                                    # cho tap test (KHONG fit lai!)

    model = LinearRegression()
    model.fit(Xp_train, y_train)

    train_mse = mean_squared_error(y_train, model.predict(Xp_train))
    test_mse = mean_squared_error(y_test, model.predict(Xp_test))
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)

    print(f"bac {degree:2d}: MSE train = {train_mse:8.2f}   |   MSE test = {test_mse:9.2f}")

# GIẢI THÍCH ĐIỂM CỰC KỲ QUAN TRỌNG TRONG CODE: poly.fit_transform() dùng
# CHO TẬP TRAIN, nhưng poly.transform() (KHÔNG có "fit_") dùng cho tập test.
# "fit" nghĩa là HỌC ra công thức biến đổi TỪ dữ liệu; nếu lỡ gọi fit_
# transform() trên CẢ tập test, mô hình sẽ "nhìn thấy" thống kê của tập test
# trong lúc lẽ ra không được biết gì về nó -> gọi là "data leakage" (rò rỉ dữ
# liệu) -> đánh giá cuối cùng sẽ LẠC QUAN GIẢ TẠO, không phản ánh đúng khả
# năng thật của mô hình trên dữ liệu hoàn toàn mới. Đây LÀ MỘT LỖI RẤT PHỔ
# BIẾN của người mới học ML, cần ghi nhớ kỹ: BẤT KỲ bước tiền xử lý nào "học"
# từ dữ liệu (StandardScaler ở Phần 1.16 cũng vậy) đều PHẢI fit trên TRAIN,
# rồi chỉ transform (không fit lại) trên TEST.
#
# GIẢI THÍCH KẾT QUẢ THẬT (đây chính là hiện tượng overfitting, nhìn thấy rõ
# bằng số liệu thật, không phải lý thuyết suông):
#
#   bac  1: MSE train=3854.11   MSE test=4061.83     )
#   bac  2: MSE train=3849.33   MSE test=4085.03     ) UNDERFIT / VUA PHAI:
#   bac  3: MSE train=3848.45   MSE test=4064.44     ) train va test gan
#   bac  5: MSE train=3823.76   MSE test=4085.85     ) nhau, ca 2 deu on dinh
#   bac  8: MSE train=3735.15   MSE test=7875.32     <- test BAT DAU vot len!
#   bac 12: MSE train=3708.17   MSE test=29988.10    <- OVERFIT RO RANG!
#
# QUAN SÁT MẤU CHỐT: MSE trên TẬP TRAIN liên tục giảm nhẹ dần khi bậc tăng
# (3854 -> 3849 -> 3848 -> 3824 -> 3735 -> 3708) - đúng như dự đoán, mô hình
# CÀNG PHỨC TẠP càng "khớp" tốt hơn với chính dữ liệu nó ĐÃ THẤY. NHƯNG MSE
# trên TẬP TEST lại đi theo hướng NGƯỢC LẠI từ bậc 8 trở đi: nhảy vọt từ
# ~4085 (bậc 5) lên 7875 (bậc 8), rồi lên tới 29988 (bậc 12) - một con số
# TỆ HƠN RẤT NHIỀU so với chỉ dùng đường thẳng bậc 1 đơn giản (4061.83)!
#
# ĐÂY CHÍNH LÀ OVERFITTING BẰNG SỐ LIỆU THẬT: đa thức bậc 12 "vặn vẹo" đường
# cong để đi SÁT qua từng điểm dữ liệu train (kể cả nhiễu ngẫu nhiên của
# từng bệnh nhân cụ thể), nhưng đường cong "vặn vẹo" đó không còn phản ánh
# QUY LUẬT CHUNG nữa, nên khi gặp bệnh nhân MỚI (tập test), nó dự đoán rất
# TỆ. Đây đúng là ẩn dụ "học tủ" ở Phần 0.4: bậc 12 gần như "học thuộc lòng"
# 353 điểm train, mất khả năng "khái quát hoá" (generalize) cho dữ liệu mới.

# %%
# TRỰC QUAN HOÁ: so sánh 4 đường cong (bậc 1, 3, 8, 12) trên cùng dữ liệu thật
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
degrees_to_plot = [1, 3, 8, 12]
x_smooth = np.linspace(x_bmi_train.min(), x_bmi_train.max(), 200).reshape(-1, 1)

for ax, degree in zip(axes.flat, degrees_to_plot):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp_train = poly.fit_transform(x_bmi_train.reshape(-1, 1))
    model = LinearRegression().fit(Xp_train, y_train)

    y_smooth = model.predict(poly.transform(x_smooth))

    ax.scatter(x_bmi_train, y_train, alpha=0.3, s=15, color='cornflowerblue', label='du lieu train')
    ax.plot(x_smooth, y_smooth, color='crimson', linewidth=2, label=f'da thuc bac {degree}')
    ax.set_ylim(y_train.min()-50, y_train.max()+50)     # co dinh truc y de de so sanh
    idx_in_full_list = degrees_to_try.index(degree)
    ax.set_title(f'Bac {degree}  (train_mse={train_mse_list[idx_in_full_list]:.0f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Overfitting minh hoa bang du lieu THAT: bac cang cao, duong cong cang "van veo"')
plt.tight_layout()
plt.show()

# GIẢI THÍCH HÌNH VẼ: nhìn 4 ô, bậc 1 là đường thẳng đơn giản; bậc 3 hơi cong
# nhẹ; bậc 8 và đặc biệt bậc 12 sẽ có những đoạn "uốn lượn" bất thường ở
# gần rìa dữ liệu (nơi có ít điểm train) - đó chính là mô hình đang cố "vặn"
# đường cong để đi qua các điểm nhiễu ngẫu nhiên, thay vì theo xu hướng
# chung. Trục y đã CỐ ĐỊNH giống nhau ở cả 4 ô để thấy rõ độ "vặn vẹo" tăng
# dần.
#
# TÓM TẮT 1 CÂU: overfitting THẬT SỰ xảy ra khi mô hình quá phức tạp so với
# lượng dữ liệu - train error tiếp tục giảm nhưng test error lại TĂNG - và
# CÁCH DUY NHẤT để phát hiện là LUÔN so sánh train vs test, không bao giờ chỉ
# nhìn train.

# %%
# ==============================================================================
# PHẦN 1.14 - REGULARIZATION (DIEU CHUAN HOA): RIDGE REGRESSION (L2)
# ==============================================================================
#
# Phần 1.13 vừa cho thấy đa thức bậc cao (bậc 8, 12) bị overfitting nặng.
# Regularization là một nhóm kỹ thuật giúp GIẢM overfitting MÀ VẪN GIỮ mô
# hình phức tạp (không phải quay lại bậc 1 đơn giản) - bằng cách THÊM một
# "hình phạt" (penalty) vào hàm mất mát, buộc các trọng số w KHÔNG ĐƯỢC quá
# lớn.
#
# TẠI SAO TRỌNG SỐ LỚN LẠI LIÊN QUAN ĐẾN OVERFITTING? Hãy nhìn lại bộ trọng
# số "Plain" (không regularization) ở đa thức bậc 8 ngay dưới đây - một số hệ
# số sẽ LỚN TỚI HÀNG NGHÌN. Trọng số CÀNG LỚN, đường cong CÀNG NHẠY (dốc)
# với thay đổi nhỏ của x -> đường cong càng dễ "vặn vẹo" đột ngột như đã thấy
# ở Phần 1.13. Ép trọng số nhỏ lại -> đường cong "mượt" hơn, ít bị vặn vẹo
# theo nhiễu.
#
# CÔNG THỨC RIDGE (L2 Regularization):
#
#     J_ridge(W, b) = (1/2m)*SUM(y_hat_i - y_i)^2  +  lambda * SUM(w_j^2)
#                     |______________________________|   |______________|
#                          MSE nhu binh thuong              PENALTY MOI:
#                                                          tong BINH PHUONG
#                                                          cua TAT CA trong so
#
# "lambda" (viet la alpha trong sklearn - KHONG lien quan gi den learning
# rate alpha o Phan 1.5, chi TRUNG TEN goi bien, can phan biet ro) la 1
# HYPERPARAMETER (nguoi tu chon) quyet dinh PHAT NANG hay NHE:
#   lambda = 0        -> quay lai y het Linear Regression thuong (khong phat)
#   lambda cang LON    -> cang ep trong so w VE GAN 0 -> mo hinh cang DON GIAN
#   lambda -> vo cuc   -> moi trong so w deu bi ep ve 0, mo hinh chi con lai b
#                         (du doan hang so - underfitting hoan toan)
#
# ĐIỀU GÌ "VỠ" NẾU KHÔNG CÓ REGULARIZATION khi mô hình quá phức tạp (như đa
# thức bậc 8, 12)? Đã THẤY rõ ở Phần 1.13: test MSE tăng vọt lên 7875 (bậc 8)
# hoặc 29988 (bậc 12). Regularization là công cụ TRỰC TIẾP giải quyết đúng
# vấn đề này mà KHÔNG cần giảm bậc đa thức.
#
# QUAN TRỌNG: PHẢI CHUẨN HOÁ (SCALE) FEATURE TRƯỚC KHI DÙNG REGULARIZATION.
# Lý do: nếu 2 feature có THANG ĐO khác nhau (vd: x^2 có gia trị nho hon
# nhieu so voi x, do x da chuan hoa nho hon 1), phần phạt lambda*w^2 sẽ tác
# động KHÔNG CÔNG BẰNG lên các feature khác thang đo - feature nào có giá trị
# gốc NHỎ sẽ "cần" trọng số w LỚN hơn để có cùng mức ảnh hưởng, nhưng lại bị
# regularization phạt náng hơn OAN UỔNG chỉ vì thang đo, không phải vì nó
# thực sự "quan trọng" hay "không quan trọng".

poly8 = PolynomialFeatures(degree=8, include_bias=False)
Xp8_train = poly8.fit_transform(x_bmi_train.reshape(-1, 1))
Xp8_test = poly8.transform(x_bmi_test.reshape(-1, 1))

scaler_poly = StandardScaler()                                  # se hoc ky ve StandardScaler o Phan 1.16
Xp8_train_scaled = scaler_poly.fit_transform(Xp8_train)         # fit TREN TRAIN
Xp8_test_scaled = scaler_poly.transform(Xp8_test)                # chi transform tren TEST

model_plain = LinearRegression().fit(Xp8_train_scaled, y_train)
model_ridge = Ridge(alpha=1.0).fit(Xp8_train_scaled, y_train)   # luu y: sklearn dat ten
                                                                    # tham so lambda LA "alpha"
                                                                    # (khac alpha cua Gradient
                                                                    # Descent - chi trung ten!)

print("He so (8 he so, ung voi x, x^2, ..., x^8) - KHONG regularization:")
print(model_plain.coef_.round(2))
print("He so - CO Ridge (lambda=1.0):")
print(model_ridge.coef_.round(2))
print()
print(f"Plain -> train MSE = {mean_squared_error(y_train, model_plain.predict(Xp8_train_scaled)):.2f}"
      f" | test MSE = {mean_squared_error(y_test, model_plain.predict(Xp8_test_scaled)):.2f}")
print(f"Ridge -> train MSE = {mean_squared_error(y_train, model_ridge.predict(Xp8_train_scaled)):.2f}"
      f" | test MSE = {mean_squared_error(y_test, model_ridge.predict(Xp8_test_scaled)):.2f}")

# GIẢI THÍCH SỐ LIỆU THẬT:
#   He so Plain : [88.39, -0.13, -294.59, 0.65, 1125.69, -331.4, -1704.28, 1168.94]
#   He so Ridge : [53.08, -8.87,  -17.98, 31.0,    4.75,  -5.07,   -9.64,    1.27]
#
# TẤT CẢ hệ số Ridge đều CÓ ĐỘ LỚN NHỎ HƠN NHIỀU so với Plain (vd: hệ số thứ
# 7 giảm từ -1704.28 xuống chỉ còn -9.64) - đúng như lý thuyết dự đoán: Ridge
# "ép co lại" MỌI trọng số, không có trọng số nào bị đưa về CHÍNH XÁC 0 (đây
# là điểm khác biệt quan trọng với Lasso ở Phần 1.15 ngay sau đây).
#
#   Plain: train=3735.15  test=7875.32   (khoang cach train-test: 4140.17 -
#          RAT LON, dau hieu overfitting ro rang, giong het Phan 1.13)
#   Ridge: train=3821.88  test=4036.68   (khoang cach: chi 214.80 - GAN
#          NHAU HON NHIEU)
#
# Ridge làm train MSE hơi TỆ ĐI một chút (3735 -> 3821, vì mô hình bị "trói
# tay" không được khớp quá sát dữ liệu train nữa), NHƯNG đổi lại test MSE
# TỐT LÊN RẤT NHIỀU (7875 -> 4036, giảm gần MỘT NỬA!) - đây CHÍNH LÀ đánh đổi
# regularization mang lại: hy sinh một chút độ khớp trên train, để đổi lấy
# khả năng KHÁI QUÁT HOÁ (generalize) tốt hơn nhiều trên dữ liệu mới.


# %%
# ==============================================================================
# PHẦN 1.15 - REGULARIZATION: LASSO REGRESSION (L1) VA SO SANH VOI RIDGE
# ==============================================================================
#
# Lasso GIỐNG Ridge ở Ý TƯỞNG chung (thêm hình phạt để ép trọng số nhỏ lại),
# nhưng KHÁC ở CÁCH TÍNH hình phạt: dùng GIÁ TRỊ TUYỆT ĐỐI thay vì bình
# phương:
#
#     J_lasso(W, b) = (1/2m)*SUM(y_hat_i - y_i)^2  +  lambda * SUM(|w_j|)
#                                                                  ^^^^^^^
#                                                            GIA TRI TUYET DOI
#                                                            (khong phai binh
#                                                             phuong nhu Ridge)

model_lasso = Lasso(alpha=1.0).fit(Xp8_train_scaled, y_train)

print("He so - CO Lasso (lambda=1.0):")
print(model_lasso.coef_.round(2))
print()
n_zero_lasso = int(np.sum(model_lasso.coef_ == 0))
n_zero_ridge = int(np.sum(model_ridge.coef_ == 0))
print(f"So he so BANG 0 tuyet doi - Ridge: {n_zero_ridge}/8   |   Lasso: {n_zero_lasso}/8")
print(f"Lasso -> train MSE = {mean_squared_error(y_train, model_lasso.predict(Xp8_train_scaled)):.2f}"
      f" | test MSE = {mean_squared_error(y_test, model_lasso.predict(Xp8_test_scaled)):.2f}")

# GIẢI THÍCH SỐ LIỆU THẬT - ĐIỂM KHÁC BIỆT MẤU CHỐT GIỮA RIDGE VÀ LASSO:
#   He so Lasso: [45.12, 0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0]
#   -> 6 TRONG SO 8 he so bi dua ve DUNG BANG 0 (khong phai "gan 0" nhu
#   Ridge, ma la CHINH XAC 0.0)! Chi con lai he so ung voi x^1 (45.12) va x^4
#   (2.2) la khac 0.
#
# TẠI SAO LASSO CÓ THỂ ĐƯA VỀ ĐÚNG 0 CÒN RIDGE THÌ KHÔNG (giải thích trực
# giác hình học, không cần chứng minh chặt chẽ): hình phạt Ridge (tổng bình
# phương) tạo ra một "đường bao" hình TRÒN/ELLIPSE quanh gốc toạ độ trong
# không gian trọng số; hình phạt Lasso (tổng trị tuyệt đối) tạo ra một
# "đường bao" hình KIM CƯƠNG (có GÓC NHỌN) tại các trục toạ độ. Điểm tối ưu
# của bài toán tối ưu có ràng buộc RẤT DỄ "rơi đúng vào" một trong các góc
# nhọn đó của hình kim cương (nơi một hoặc nhiều toạ độ = 0), trong khi hình
# tròn của Ridge KHÔNG CÓ góc nhọn nào để "rơi vào" một cách đặc biệt như
# vậy.
#
# Ý NGHĨA THỰC TIỄN CỰC KỲ QUAN TRỌNG: Lasso có thể dùng để LÀM FEATURE
# SELECTION TỰ ĐỘNG (tự động loại bỏ feature không quan trọng bằng cách đưa
# hệ số về đúng 0) - đây là một công cụ RẤT hữu ích trong Data Science thực
# tế khi có hàng trăm feature và muốn tự động tìm ra tập con quan trọng nhất.
#
# ------------------------------------------------------------------------------
# BẢNG SO SÁNH TỔNG KẾT: PLAIN vs RIDGE (L2) vs LASSO (L1)
# ------------------------------------------------------------------------------
#  Tieu chi          | Plain (khong RL) | Ridge (L2)       | Lasso (L1)
#  -------------------|-------------------|------------------|------------------
#  Cong thuc phat     | khong co         | lambda*sum(w^2)  | lambda*sum(|w|)
#  Train MSE (that)   | 3735.15          | 3821.88          | 3848.97
#  Test MSE (that)    | 7875.32          | 4036.68          | 4093.48
#  He so ve dung 0?   | khong            | KHONG (chi nho)  | CO (6/8 he so)
#  Dung khi nao        | it feature,      | nhieu feature    | muon TU DONG
#                      | it lo overfit    | tuong quan cao   | chon loc feature
#                      |                  | (giu tat ca lai) | (loai bot feature)
#
# TÓM TẮT 1 CÂU PHẦN 1.14-1.15: Regularization thêm hình phạt vào hàm mất
# mát để ép trọng số nhỏ lại, đổi một chút độ khớp trên train lấy khả năng
# khái quát hoá tốt hơn NHIỀU trên test - Ridge co nho đều tất cả hệ số,
# Lasso có thể đưa hẳn một số hệ số về 0 để tự động chọn feature.

# %%
# ==============================================================================
# PHẦN 1.16 - FEATURE SCALING: TAI SAO CAN, MINH HOA BANG DU LIEU THAT KHAC
# ==============================================================================
#
# Xuyên suốt file này, feature "bmi" và các feature khác trong diabetes
# dataset đã được sklearn CHUẨN HOÁ SẴN (nhớ lại Phần 1.0/1.1: std=0.0476
# cho MỌI cột) - nên ta CHƯA THẬT SỰ thấy vấn đề "feature không cùng thang
# đo" trông như thế nào trên dữ liệu THẬT chưa qua xử lý. Để thấy rõ, ta mượn
# tạm bộ dữ liệu THẬT thứ 2: breast cancer dataset (569 ca chẩn đoán ung thư
# vú thật, cũng có sẵn trong sklearn, không cần tải mạng) - đây SẼ LÀ nhân
# vật chính của Phần 2 (Classification), nên xem trước ở đây cũng là một
# cách "làm quen" sớm.

bc_data = load_breast_cancer()
df_bc = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)

print("Shape:", bc_data.data.shape, " (so ca chan doan x so chi so do)")
print("2 nhan: ", bc_data.target_names, " (0=malignant/ac tinh, 1=benign/lanh tinh)")
print()
print("Thong ke 4 chi so, CHUA chuan hoa:")
print(df_bc[['mean radius', 'mean area', 'mean smoothness', 'mean compactness']].describe().round(3))

# GIẢI THÍCH SỐ LIỆU THẬT - VẤN ĐỀ THANG ĐO (SCALE) LỘ RÕ:
#   mean radius      : trung binh=14.127,  do rong [6.981  , 28.11]
#   mean area        : trung binh=654.889, do rong [143.5  , 2501.0]
#   mean smoothness  : trung binh=0.096,   do rong [0.053  , 0.163]
#   mean compactness : trung binh=0.104,   do rong [0.019  , 0.345]
#
# "mean area" (diện tích khối u) dao động tới HÀNG NGHÌN đơn vị (143 đến
# 2501), trong khi "mean smoothness" (độ trơn nhẵn) chỉ dao động QUANH 0.05
# đến 0.16 - CHÊNH LỆCH THANG ĐO tới CẢ CHỤC NGHÌN LẦN giữa 2 feature THẬT
# trong CÙNG một bộ dữ liệu y tế!
#
# HẬU QUẢ NẾU CHẠY GRADIENT DESCENT TRỰC TIẾP TRÊN DỮ LIỆU THÔ NÀY (không
# scale): nhớ lại hiện tượng đã thấy ở Phần 1.5 - feature nào có PHƯƠNG SAI
# (variance) lớn hơn sẽ tạo ra hướng "RẤT DỐC" trong hàm J, còn feature
# phương sai nhỏ tạo hướng "RẤT THOẢI". Với chênh lệch thang đo LỚN NHƯ THẾ
# NÀY (area vs smoothness), hàm J sẽ có dạng một "thung lũng CỰC KỲ dẹt và
# dài" - Gradient Descent sẽ "zic zac" (dao động qua lại) rất mạnh theo
# hướng dốc (area) trong khi TIẾN RẤT CHẬM theo hướng thoải (smoothness) ->
# cần RẤT NHIỀU vòng lặp mới hội tụ, hoặc phải chọn alpha CỰC NHỎ (an toàn
# cho hướng dốc nhất) khiến hướng thoải gần như KHÔNG di chuyển được bao
# nhiêu trong số vòng lặp cho phép.
#
# GIẢI PHÁP: STANDARDSCALER (chuẩn hoá Z-score) - công thức, cho MỖI feature
# RIÊNG BIỆT:
#
#     x_scaled = (x - mean(x)) / std(x)
#
# Sau phép biến đổi này, MỌI feature đều có trung bình = 0 va do lech chuan
# = 1, bất kể đơn vị GỐC của nó là gì (mm, mm^2, hay ty le khong don vi).

scaler_demo = StandardScaler()
cols_demo = ['mean radius', 'mean area', 'mean smoothness', 'mean compactness']
scaled_demo = scaler_demo.fit_transform(df_bc[cols_demo])

print("\nSau StandardScaler - trung binh (phai ~0) va do lech chuan (phai ~1):")
print("Trung binh:", scaled_demo.mean(axis=0).round(4))
print("Do lech chuan:", scaled_demo.std(axis=0).round(4))

# GIẢI THÍCH: sau khi scale, CẢ 4 cột đều có trung bình xấp xỉ 0.0000 và độ
# lệch chuẩn CHÍNH XÁC bằng 1.0000 - dù đơn vị GỐC của chúng khác nhau hoàn
# toàn (mm cho "radius", mm^2 cho "area", tỉ lệ không đơn vị cho
# "smoothness"). Bây giờ TẤT CẢ feature đều ở CÙNG một thang đo công bằng,
# nên Gradient Descent sẽ hội tụ NHANH VÀ ỔN ĐỊNH hơn nhiều (đây chính là lý
# do Phần 1.14 - Regularization - đã BẮT BUỘC scale trước khi áp dụng Ridge/
# Lasso, vì hình phạt lambda*w^2 hoặc lambda*|w| CŨNG bị ảnh hưởng bởi thang
# đo y hệt Gradient Descent).
#
# LƯU Ý QUAN TRỌNG (nhắc lại nguyên tắc đã học ở Phần 1.13): StandardScaler
# PHẢI được fit() TRÊN TẬP TRAIN, rồi chỉ transform() (không fit lại) trên
# tập test - nếu không sẽ bị "data leakage" y hệt lỗi đã cảnh báo ở Phần
# 1.13 với PolynomialFeatures.
#
# SO SÁNH: KHI NÀO CẦN SCALING, KHI NÀO KHÔNG BẮT BUỘC?
#   CẦN scaling             : Gradient Descent (mọi mô hình dùng nó, bao gồm
#                              Neural Network ở Phần 7), Regularization
#                              (Ridge/Lasso), KNN va SVM (Phan 2 - dua tren
#                              KHOANG CACH giua cac diem, thang do anh huong
#                              truc tiep den khoang cach), PCA (Phan 4).
#   KHÔNG bắt buộc scaling   : Normal Equation (Phần 1.9 - vẫn ra đúng đáp
#                              án dù không scale, chỉ là các hệ số w_i sẽ có
#                              độ lớn khác nhau tương ứng thang đo gốc), cây
#                              quyết định/Decision Tree và các mô hình dựa
#                              trên cây (Phần 2-3 - chúng chỉ so sánh "lớn
#                              hơn/nhỏ hơn 1 ngưỡng" trên TỪNG feature riêng
#                              lẻ, không bị ảnh hưởng bởi thang đo TƯƠNG ĐỐI
#                              giữa các feature khác nhau).
#
# TÓM TẮT 1 CÂU: Feature Scaling đưa mọi feature về CÙNG một thang đo (trung
# bình 0, độ lệch chuẩn 1) để Gradient Descent hội tụ nhanh và ổn định, và
# để Regularization phạt các trọng số một cách CÔNG BẰNG - dữ liệu breast
# cancer thật vừa cho thấy chênh lệch thang đo có thể lên tới HÀNG NGHÌN LẦN
# giữa các feature y tế thực tế.

# %%
# ==============================================================================
# TONG KET PHAN 1 - NHUNG GI DA HOC DUOC (VA CHUAN BI CHO PHAN 2)
# ==============================================================================
#
# ------------------------------------------------------------------------------
# BẢN ĐỒ TƯ DUY TOÀN BỘ PHẦN 0 + PHẦN 1 (đọc lại 1 lượt để hệ thống hoá)
# ------------------------------------------------------------------------------
#  PHẦN 0 - NỀN TẢNG
#    ML = máy tự học quy luật từ dữ liệu (thay vì người viết sẵn luật)
#    Supervised (có y) chia thành Regression (y liên tục) / Classification
#    (y rời rạc); Unsupervised (không có y) tìm cấu trúc ẩn
#    Quy trình: Thu thập -> EDA -> Tiền xử lý -> Chọn mô hình -> Huấn luyện
#    -> Đánh giá -> Tinh chỉnh
#    Overfitting (học vẹt, tốt trên train tệ trên test) vs Underfitting (dở
#    cả 2)
#
#  PHẦN 1 - LINEAR REGRESSION (tất cả đã CHẠY THẬT, KHÔNG chỉ lý thuyết)
#    1. y_hat = w*x + b (don bien) hoac y_hat = W.X + b (da bien)
#    2. Cost function MSE do "do te" cua (w,b): J = (1/2m)*SUM(y_hat-y)^2
#    3. Gradient Descent: lap w := w - alpha*dJ/dw, hoi tu ve DUNG dap an
#       sklearn khi chay du vong lap (da kiem chung: 998.5777 = 998.5777)
#    4. Normal Equation: cong thuc dai so tuyen tinh cho nghiem THANG, khong
#       can lap (Theta = (X^T X)^-1 X^T y)
#    5. Multicollinearity: feature tuong quan cao (s1-s2: r=0.89) lam GD
#       hoi tu cham va he so KHONG on dinh, nhung DU DOAN van dang tin cay
#       (R2 GD=0.4552 rat gan R2 Normal Eq=0.4526)
#    6. Danh gia: MSE (huan luyen), RMSE/MAE (bao cao, don vi that), R2
#       (chuan hoa 0-1, so voi baseline doan trung binh) - mo hinh 10 feature
#       dat R2=0.4526 tren 89 benh nhan CHUA TUNG thay
#    7. Overfitting THAT: da thuc bac 12 co train MSE tot nhat (3708) nhung
#       test MSE TE NHAT (29988) - CANG PHUC TAP KHONG DONG NGHIA CANG TOT
#    8. Regularization: Ridge (L2, co nho deu) va Lasso (L1, dua ve dung 0 -
#       tu dong chon feature) deu giam test MSE cua da thuc bac 8 tu 7875
#       xuong con ~4036-4093, GAN VOI muc bac 1 (4061) nhung GIU duoc do
#       phuc tap can thiet
#    9. Feature Scaling: du lieu breast cancer that cho thay chenh lech thang
#       do co the toi HANG NGHIN LAN giua cac feature y te that -
#       StandardScaler dua ve cung thang do (mean=0, std=1)
#
# ------------------------------------------------------------------------------
# NHUNG CON SO "CHUAN" CUA PHAN NAY (dung de SO SANH khi hoc cac mo hinh
# manh hon o Phan 2, 3 - "manh hon Linear Regression bao nhieu" se duoc do
# bang chinh nhung con so nay lam moc)
# ------------------------------------------------------------------------------
print("="*60)
print("BANG DIEM CHUAN - DIABETES DATASET (10 feature, test set 89 benh nhan)")
print("="*60)
print(f"{'Mo hinh':<30}{'Test MSE':>12}{'Test R2':>10}")
print(f"{'Linear Regression (full)':<30}{mse_ne:>12.2f}{r2_ne:>10.4f}")
print("-"*52)
print("=> Day la 'diem xuat phat' - Phan 2 (Classification) va Phan 3")
print("   (Ensemble - Random Forest, Gradient Boosting...) se dung cac ky")
print("   thuat MANH HON de xem co the vuot qua con so R2=0.4526 nay hay")
print("   khong, tren cung 1 loai bai toan du doan.")


# %%
# ==============================================================================
# XEM TRUOC PHAN 2/8: PHAN LOAI (CLASSIFICATION)
# ==============================================================================
#
# Phần 2 (file kế tiếp) sẽ dùng CHÍNH bộ dữ liệu breast cancer thật vừa xem
# thoáng qua ở Phần 1.16 (569 ca, 30 chỉ số, nhãn malignant/benign) để học:
#
#   - Logistic Regression: về bản chất là Linear Regression + 1 hàm "ép"
#     kết quả về khoảng (0,1) để diễn giải thành XÁC SUẤT - hiểu Phần 1 kỹ
#     sẽ thấy Logistic Regression KHÔNG có gì mới về mặt cost function/
#     gradient descent, chỉ khác ở công thức y_hat.
#   - K-Nearest Neighbors (KNN): dự đoán dựa trên "hàng xóm gần nhất" - CHÍNH
#     LÝ DO cần Feature Scaling (Phần 1.16) sẽ càng RÕ hơn ở đây, vì KNN đo
#     "gần" bằng khoảng cách Euclid, thang đo lệch sẽ làm khoảng cách bị SAI
#     LỆCH hoàn toàn theo feature có giá trị lớn nhất.
#   - Decision Tree, SVM, Naive Bayes
#   - Confusion Matrix, Precision, Recall, F1-score, ROC-AUC (bộ chỉ số đánh
#     giá RIÊNG cho bài toán phân loại, khác hẳn MSE/RMSE/R2 vừa học)
#
# File 1/8 đến đây là TRỌN VẸN. Khi đã chạy thử, đọc kỹ, và cảm thấy VỮNG với
# toàn bộ Phần 0 + Phần 1, nhắn lại để tiếp tục xây File 2/8 (Phần 2 -
# Classification) với cùng chuẩn: dataset thật, annotate từng dòng, không
# cắt góc.
#
# ==============================================================================
#                                HET FILE 1/8
# ==============================================================================
