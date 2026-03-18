import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.isnull().sum())
df.columns = df.columns.str.lower()
# print(df.info())

# sns.barplot(data=df, x='churn', y='monthlycharges')


# for col in cat_cols:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x=col, hue='churn', data=df)
    # plt.show()

ct = pd.crosstab(df['tenure'], df['churn'], normalize='index') * 100
ct.plot(kind='bar', stacked=True)
# plt.show()

df = df.drop(['gender', 'customerid', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'paperlessbilling', 'paymentmethod', 'seniorcitizen'], axis=1)
cat_cols = df.select_dtypes(include='object', exclude=['int64', 'float64']).columns
cat_cols = cat_cols.drop(['churn', 'totalcharges'])
print(cat_cols)

# df = pd.get_dummies(df, columns=['internetservice'])

# for col in cat_cols:
#     print(f"{col} : "+df[col].unique())

# print(df['onlinebackup'].unique())
# print(df.head())