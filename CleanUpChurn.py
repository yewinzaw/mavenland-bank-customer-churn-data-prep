import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 50)
pd.set_option('display.expand_frame_repr', False)
churn_messy_excel_file_path = "Dataset/Bank_Churn_Messy.xlsx"

# region *** Step 1: Clean up: Account Info ***
account_info = pd.read_excel(churn_messy_excel_file_path, sheet_name="Account_Info")

# 1.1 Remove duplicated rows: CustomerId
account_info = account_info.drop_duplicates("CustomerId")

# 1.2 Remove currency symbol and non-digits and convert to float
account_info["Balance"] = account_info["Balance"].str.replace(r'[^\d.-]', '', regex=True).astype(float)

# 1.3 Remove HasCrCard column if it is identical to IsActiveMember
#  Removing redundant features prevents perfect multicollinearity in ML models
if account_info["HasCrCard"].equals(account_info["IsActiveMember"]):
    account_info = account_info.drop(columns=["HasCrCard"])

# 1.4 Convert Yes/No to 1/0 (case-insensitive, trims spaces)
account_info["IsActiveMember"] = (account_info["IsActiveMember"].str.strip().str.lower() == "yes").astype(int)

# endregion
# region *** Step 2: Clean up: Customer Info ***
customer_info = pd.read_excel(churn_messy_excel_file_path, sheet_name="Customer_Info")

# 2.1 Remove duplicated CustomerId
customer_info = customer_info.drop_duplicates("CustomerId")

# 2.2 Convert Gender to 1/0 (Male=1, Female=0) (case-insensitive, trims spaces)
customer_info["Gender"] = customer_info["Gender"].str.strip().str.lower().eq("male").astype(int)

# 2.3 Remove currency symbol and non-digits and convert to float
customer_info["EstimatedSalary"] = customer_info["EstimatedSalary"].str.replace(r'[^\d.-]', '', regex=True).astype(float)

# 2.4 Impute missing values using median
customer_info = customer_info.fillna({"Age": customer_info["Age"].median(), "Surname": "Unknown"})
customer_info["EstimatedSalary"] = customer_info["EstimatedSalary"].replace(-999999.0, customer_info["EstimatedSalary"].median())
customer_info["Geography"] = customer_info["Geography"].str.strip().replace({"FRA": "France", "French": "France"})

# endregion
# region *** Step 3: Left Join ***

# 3.1 Left join: keep all customers (host = Customer_Info)
merged_df = customer_info.merge(account_info, how="left", on="CustomerId")

# 3.2 Check NaN introduced by left join
# print(merged_df.isna().sum())

# 3.3 Remove duplicate Tenure columns if identical
if merged_df["Tenure_x"].equals(merged_df["Tenure_y"]):
    merged_df = merged_df.drop(columns=["Tenure_y"]).rename(columns={"Tenure_x": "Tenure"})

# endregion
# region *** Step 4: Charts ***

# Build a bar chart displaying the count of churners (Exited=1) vs. non-churners (Exited=0)
merged_df["Exited"].value_counts(normalize=True).plot.bar(
    title="Proportion of Exited Customers",
    xlabel="Status (0 = Retained, 1 = Exited)",
    ylabel="",
    rot=0  # Keeps the x-axis text horizontal
)
plt.show()

# Explore the categorical variables vs. the target, and look at the percentage of Churners by Geography and Gender
for col, x_label in [("Geography", ""), ("Gender", "Gender (0= Male, 1=Female)")]:
    sns.barplot(x=col, y="Exited", data=merged_df)
    plt.ylabel("")
    plt.xlabel(x_label)
    plt.title(f"Churn Rate by {col}")
    plt.show()

# Build box plots for each numeric field, broken out by churners vs. non-churners
for col in ["CreditScore", "Age", "Tenure", "EstimatedSalary", "Balance"]:
    sns.boxplot(y=col, x="Exited", data=merged_df, hue="Exited", legend=False)
    plt.xlabel("Status (0 = Retained, 1 = Exited)")
    plt.ylabel("")
    plt.title(f"Churn Rate by {col}")
    plt.show()

# Build histograms for each numeric field, broken out by churners vs. non-churners
for col in ["CreditScore", "Age", "Tenure", "EstimatedSalary", "Balance"]:
    sns.histplot(y=col, data=merged_df, hue="Exited", kde=True, legend=True)
    plt.xlabel("Status (0 = Retained, 1 = Exited)")
    plt.ylabel("")
    plt.title(f"Churn Rate by {col}")
    plt.show()

# endregion
# region *** Step 5: Build ML Model ***

# 5.1 Exclude unsuitable fields
model_df = merged_df.drop(columns=["CustomerId", "Surname"])

# 5.2 One-hot encode Geography
model_df = pd.get_dummies(model_df, columns=["Geography"], drop_first=True, dtype=int)

# Task: Create a new “balance_v_income” feature, which divides a customer’s bank balance by their estimated salary, then visualize that feature vs. churn status

# 5.3 Create ratio feature
model_df["balance_v_income"] = model_df["Balance"] / model_df["EstimatedSalary"]

# 5.4 Filter out extreme values (95th percentile)
upper_cutoff = model_df["balance_v_income"].quantile(0.95)
model_cutoff = model_df[model_df["balance_v_income"] < 10]
print("Rows before:", len(model_df), "Rows after:", len(model_cutoff))

# 5.5 Visualization: balance_v_income vs churn
sns.boxplot(x="Exited", y="balance_v_income", data=model_cutoff,hue="Exited", legend=False)
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Balance / EstimatedSalary")
plt.show()

sns.barplot(x="Exited",y="balance_v_income", data=model_cutoff, hue="Exited")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Balance / EstimatedSalary")
plt.show()
# endregion
