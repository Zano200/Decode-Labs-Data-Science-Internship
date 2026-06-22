import pandas as pd
import numpy as np
import pandera as pa
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# 1: LOAD DATA

df = pd.read_csv("Dataset for Data Analytics.csv")
print("Shape:", df.shape)
print(df.head(3))
print(df.dtypes)

# 2: MISSING DATA ANALYSIS
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_pct})
print(missing_report[missing_report["Missing Count"] > 0])

# 3: HANDLE MISSING VALUES
df["CouponCode"] = df["CouponCode"].fillna("No Coupon")
print("After imputation:")
print(df.isnull().sum())

# 4: OUTLIER DETECTION AND WINSORIZATION
numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    print(col + ": " + str(outliers_before) + " outliers capped")

# 5: FEATURE ENGINEERING
df["RevenuePerItem"] = df["TotalPrice"] / df["Quantity"]
print("Feature 1 created: RevenuePerItem")

median_price = df["TotalPrice"].median()
df["IsHighValueOrder"] = (df["TotalPrice"] > median_price).astype(int)
print("Feature 2 created: IsHighValueOrder")

df["HasCoupon"] = (df["CouponCode"] != "No Coupon").astype(int)
print("Feature 3 created: HasCoupon")

print(df[["TotalPrice", "Quantity", "RevenuePerItem", "IsHighValueOrder", "HasCoupon"]].head())

# 6: ONE-HOT ENCODING
cols_to_encode = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource"]
df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
print("Shape before encoding:", df.shape)
print("Shape after encoding:", df_encoded.shape)

# 7: COLLINEARITY CHECK
numeric_df = df[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice",
                  "RevenuePerItem", "IsHighValueOrder", "HasCoupon"]]
corr_matrix = numeric_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = [(col, row, upper_triangle.loc[row, col])
                   for col in upper_triangle.columns
                   for row in upper_triangle.index
                   if upper_triangle.loc[row, col] > 0.80]

if high_corr_pairs:
    print("High correlation pairs found (> 0.80):")
    for pair in high_corr_pairs:
        print(" ", pair[0], "and", pair[1], ":", round(pair[2], 3))
    df.drop(columns=["RevenuePerItem"], inplace=True)
    df_encoded.drop(columns=["RevenuePerItem"], inplace=True, errors="ignore")
    print("Dropped RevenuePerItem (redundant with UnitPrice).")
else:
    print("No highly collinear pairs found.")

# 8: PANDERA SCHEMA VALIDATION
schema = pa.DataFrameSchema({
    "Quantity": pa.Column(int, pa.Check.greater_than(0)),
    "UnitPrice": pa.Column(float, pa.Check.greater_than(0)),
    "TotalPrice": pa.Column(float, pa.Check.greater_than(0)),
    "IsHighValueOrder": pa.Column(int, pa.Check.isin([0, 1])),
    "HasCoupon": pa.Column(int, pa.Check.isin([0, 1])),
    "CouponCode": pa.Column(str),
})

try:
    schema.validate(df, lazy=True)
    print("Pandera validation PASSED.")
except pa.errors.SchemaErrors as e:
    print("Validation FAILED:")
    print(e.failure_cases)

# 9: SAVE CLEAN DATASET
df.to_csv("clean_dataset.csv", index=False)
print("Clean dataset saved as clean_dataset.csv")
print("Final shape:", df.shape)

# 10: VISUALISATIONS

# Chart 1: Missing Data
plt.figure(figsize=(6, 4))
plt.bar(["CouponCode"], [25.75], color="tomato")
plt.title("Missing Data Per Column (%)")
plt.ylabel("Missing %")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("viz1_missing_data.png")
plt.show()
print("Chart 1 saved")

# Chart 2: TotalPrice Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["TotalPrice"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of TotalPrice (After Winsorization)")
plt.xlabel("TotalPrice")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("viz2_totalprice_distribution.png")
plt.show()
print("Chart 2 saved")

# Chart 3: Boxplots of Numeric Columns
plt.figure(figsize=(10, 5))
df[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]].boxplot()
plt.title("Boxplots of Numeric Columns (After Winsorization)")
plt.tight_layout()
plt.savefig("viz3_boxplots.png")
plt.show()
print("Chart 3 saved")

# Chart 4: High Value Orders
plt.figure(figsize=(5, 4))
df["IsHighValueOrder"].value_counts().plot(kind="bar", color=["steelblue", "orange"])
plt.title("High Value Orders (1=Yes, 0=No)")
plt.xlabel("IsHighValueOrder")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viz4_high_value_orders.png")
plt.show()
print("Chart 4 saved")

# Chart 5: Coupon Usage
plt.figure(figsize=(5, 4))
df["HasCoupon"].value_counts().plot(kind="bar", color=["green", "grey"])
plt.title("Coupon Usage (1=Used, 0=Not Used)")
plt.xlabel("HasCoupon")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viz5_coupon_usage.png")
plt.show()
print("Chart 5 saved")

# Chart 6: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice", "IsHighValueOrder", "HasCoupon"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("viz6_correlation_heatmap.png")
plt.show()
print("Chart 6 saved")

# Chart 7: Order Count by Product
raw_df = pd.read_csv("Dataset for Data Analytics.csv")
plt.figure(figsize=(8, 4))
raw_df["Product"].value_counts().plot(kind="bar", color="mediumpurple")
plt.title("Order Count by Product")
plt.xlabel("Product")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("viz7_product_orders.png")
plt.show()
print("Chart 7 saved")

# Chart 8: Orders by Payment Method
plt.figure(figsize=(8, 4))
raw_df["PaymentMethod"].value_counts().plot(kind="bar", color="coral")
plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("viz8_payment_method.png")
plt.show()
print("Chart 8 saved")

# Chart 9: Orders by Order Status
plt.figure(figsize=(8, 4))
raw_df["OrderStatus"].value_counts().plot(kind="bar", color="teal")
plt.title("Orders by Status")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("viz9_order_status.png")
plt.show()
print("Chart 9 saved")

print("PROJECT 1 COMPLETE!")