import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, roc_auc_score
)
import pickle





df = pd.read_csv(r"C:\Users\yashk\OneDrive\Desktop\new\machine learning\project\diabetes.csv")


# print(df)


# print("Initial Data:")
# print(df.head())


cols_with_zero = ["Glucose", "BloodPressure", "BMI"]

df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)


for col in cols_with_zero:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)

# print(df.describe())


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Diabetes Dataset")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions in Diabetes Dataset", fontsize=16)
plt.tight_layout()
plt.show()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("=== Scaled Feature Sample ===")
print(X_scaled_df.head())

print("\nFeature Means After Scaling:")
print(X_scaled_df.mean().round(3))

print("\nFeature Standard Deviations After Scaling:")
print(X_scaled_df.std().round(3))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# logistics regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    return acc, cm

acc_nb, cm_nb = evaluate_model("Naïve Bayes", y_test, y_pred_nb)
acc_lr, cm_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr)



fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Naïve Bayes\nAccuracy: {acc_nb:.2f}')
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Logistic Regression\nAccuracy: {acc_lr:.2f}')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

models = {
    "Naïve Bayes": (y_pred_nb, nb.predict_proba(X_test)[:, 1]),
    "Logistic Regression": (y_pred_lr, lr.predict_proba(X_test)[:, 1])
}
results = []



for name, (y_pred, y_prob) in models.items():
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    results.append([name, precision, recall, f1, auc_score])

metrics_df = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score", "ROC-AUC"])
print("\n=== Model Performance Comparison ===")
print(metrics_df)

# === Visualize metrics ===
metrics_df.set_index("Model").plot(kind='bar', figsize=(8,6), colormap='viridis', edgecolor='black')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
for name, (y_pred, y_prob) in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.2f})")

plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# logistic regression gives an accuracy of 0.77 while naive bayes gives 0.76 , so we are using logistic regression


