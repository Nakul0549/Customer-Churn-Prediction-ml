import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle 

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.head())
# print(df.info())

df.columns = df.columns.str.strip().str.replace(" ","_").str.lower()
# print(df.info())
# print(df.isnull().sum())

df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'] = df['totalcharges'].fillna(0)
# print(df['totalcharges'].isnull().sum())

# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.tight_layout()
# plt.show()

df = df.drop('customerid', axis=1)
# print(df.info())

cat_col = df.select_dtypes(include='object').columns
cat_col = cat_col.drop('churn')


# for col in cat_col:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x=col, hue='churn', data=df)
#     plt.tight_layout()
#     plt.show()

# for col in cat_col:
#     ct = pd.crosstab(df[col], df['churn'], normalize='index')*100
#     ct.plot(kind='bar', stacked=True)
#     plt.tight_layout()
#     plt.show()

df['churn'] = df['churn'].map({"Yes":1, "No":0})
df = pd.get_dummies(df, drop_first=True)
df.columns = df.columns.str.strip().str.replace(" ","_").str.lower()
# print(df.info())
df = df.apply(lambda x: x.astype(int) if x.dtype=='bool' else x)
# print(df.info())

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test) 

lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
# print(accuracy_score(y_test, y_pred))       
# print(classification_report(y_test, y_pred))

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test, y_pred_nb)) 
# print(classification_report(y_test, y_pred_nb))

knn_model = KNeighborsClassifier(n_neighbors=5, p=2)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
# print("Accuracy: ",accuracy_score(y_test,y_pred_knn))
# print(classification_report(y_test,y_pred_knn))    

dt_model = DecisionTreeClassifier(random_state=42,criterion='entropy',max_depth=10,max_features='sqrt')
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_pred_dt))
# print(classification_report(y_test,y_pred_dt))

smv_model = SVC(class_weight='balanced',C=1,kernel='rbf',max_iter=10000, random_state=42)
smv_model.fit(X_train_scaled, y_train)
y_pred_svm = smv_model.predict(X_test_scaled)
# print("Accuracy: ",accuracy_score(y_test,y_pred_svm))
# print(classification_report(y_test,y_pred_svm))

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_clf.fit(X_train,y_train)
y_pred_rf = rf_clf.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_pred_rf))
# print(classification_report(y_test,y_pred_rf))

ada_model = AdaBoostClassifier(n_estimators=200, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_pred_ada))
# print(classification_report(y_test,y_pred_ada))

gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_pred_gb))
# print(classification_report(y_test,y_pred_gb))

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_pred_xgb))
# print(classification_report(y_test,y_pred_xgb))

pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))