# Nhật ký thực hiện dự án House Prices

## 1. Tiền xử lý dữ liệu

Kích thước ban đầu:

- Train: 1.460 dòng.
- Test: 1.459 dòng.

### Xử lý missing values

Xử lý missing values theo loại biến:

- Biến phân loại được điền bằng mode.
- Biến số được điền bằng mean.

Các nhóm biến được xử lý gồm các biến garage, basement, fireplace, masonry, frontage, diện tích tầng hầm và các biến số liên quan.

###  Xóa các cột có quá nhiều giá trị thiếu

- `Id`
- `Alley`
- `PoolQC`
- `Fence`
- `MiscFeature`

###  Loại outlier

Loại 5 quan sát bất thường có: GrLivArea > 4000, còn khoảng 1.455 dòng train.

###  One-Hot Encoding

Train và test được gộp trước khi mã hóa để bảo đảm hai tập có cùng hệ thống cột. Các biến phân loại được mã hóa bằng: pd.get_dummies(..., drop_first=True)

Sau encoding:

- Dữ liệu gộp có 2.915 dòng.
- Có 175 cột sau khi loại cột trùng.
- Dữ liệu được tách lại thành `df_train` và `df_test`.

## 2. Mô hình XGBoost

###  RandomizedSearchCV

Cấu hình tốt nhất gần tương đương:

- `learning_rate = 0.05`
- `max_depth = 3`
- `n_estimators = 1100`
- `min_child_weight = 4`
- `booster = gbtree`
- `base_score = 0.75`

###  Kết quả validation

- MAE: khoảng `14,748`.
- MSE: khoảng `501,458,754`.
- RMSE: khoảng `22,393`.
- R²: khoảng `0.904`.

XGBoost là mô hình dự đoán tốt nhất trong các mô hình đã đánh giá.


## 3. Decision Tree

Notebook ban đầu dùng: DecisionTreeClassifier()

Đây là sai vì `SalePrice` là biến liên tục.

Thay bằng: DecisionTreeRegressor(random_state=42)

###  Kết quả validation

- MAE: khoảng `26,619`.
- MSE: khoảng `1,650,635,677`.
- RMSE: khoảng `40,628`.
- R²: khoảng `0.685`.

Decision Tree kém hơn XGBoost.

## 4. Artificial Neural Network

###  Kiến trúc

Xây dựng mạng Sequential gồm các lớp Dense:

- 50 neurons.
- 25 neurons.
- 50 neurons.
- 1 neuron đầu ra.

Activation chính là `relu`, optimizer là `Adamax`, loss là RMSE tự định nghĩa.


## 5. MODEL1 hồi quy thống kê

###  Các biến sử dụng

MODEL1 sử dụng 6 biến gốc, không có biến tương tác:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`
- `YearBuilt`
- `FullBath`

Mô hình được xây dựng bằng `statsmodels.OLS`.

###  Kết quả

- R²: `0.8138`.
- Adjusted R²: `0.8130`.

Mô hình vượt yêu cầu tối thiểu 73%.

Phương trình hồi quy:

```text
SalePrice = -834583
+ 18668.2 * OverallQual
+ 61.9148 * GrLivArea
+ 11412.7 * GarageCars
+ 41.4554 * TotalBsmtSF
+ 385.336 * YearBuilt
- 9901.66 * FullBath
```

Các biến trong mô hình có p-value rất nhỏ và có ý nghĩa thống kê ở mức 5%.

###  Kiểm tra thủ công

Đối với quan sát đầu tiên:

- Giá thực tế: `$208,500`.
- Dự đoán bằng tính thủ công: `$212,304.41`.
- Dự đoán bằng statsmodels: `$212,304.41`.
- Sai số: khoảng `$3,804.41`.

Kết quả tính thủ công khớp với kết quả từ phần mềm.

###  Kiểm tra giả định

- Mô hình đạt yêu cầu về số biến và R² của bài tập.

## 6. Đánh giá MODEL2 

```text
Bias = Average(predicted - actual)
Maximum Deviation = max(abs(actual - predicted))
Mean Absolute Deviation = Average(abs(actual - predicted))
Mean Square Error = Average((actual - predicted)^2)
```

### XGBoost

- Bias: `-195.12`.
- Maximum Deviation: `124,862.56`.
- Mean Absolute Deviation: `14,748.04`.
- Mean Square Error: `501,458,753.54`.

### Decision Tree

- Bias: `4,008.37`.
- Maximum Deviation: `178,900.00`.
- Mean Absolute Deviation: `26,618.54`.
- Mean Square Error: `1,650,635,676.76`.

XGBoost tốt hơn Decision Tree ở cả MAD, Maximum Deviation và MSE.