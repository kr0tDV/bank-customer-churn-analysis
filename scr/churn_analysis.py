import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Настройки
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
output_dir = "result/churn_analysis"
os.makedirs(output_dir, exist_ok=True)

# 1. Загрузка данных
df = pd.read_csv("data/processed/Bank_Churn_cleaned.csv")
print(f"Загружено {len(df)} строк")

# 2. Общая статистика по оттоку
churn_counts = df['Exited'].value_counts().reset_index()
churn_counts.columns = ['Exited', 'Count']
churn_counts['Percentage'] = churn_counts['Count'] / len(df) * 100
print("\nРаспределение оттока:")
print(churn_counts)
churn_counts.to_csv(os.path.join(output_dir, "churn_overall.csv"), index=False)

# 3. Сравнение групп по числовым признакам
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
numeric_stats = []
for feature in numeric_features:
    for target in [0, 1]:
        data = df[df['Exited'] == target][feature]
        stats = {
            'Feature': feature,
            'Exited': target,
            'Count': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max()
        }
        numeric_stats.append(stats)
numeric_df = pd.DataFrame(numeric_stats)
print("\nЧисловые характеристики по группам:")
print(numeric_df.round(2))
numeric_df.to_csv(os.path.join(output_dir, "numeric_comparison.csv"), index=False)

# 4. Сравнение по категориальным признакам
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']
for feat in categorical_features:
    cross = pd.crosstab(df[feat], df['Exited'], normalize='index') * 100
    cross.columns = ['Stayed (%)', 'Churned (%)']
    cross = cross.reset_index()
    cross.to_csv(os.path.join(output_dir, f"crosstab_{feat}_vs_churn.csv"), index=False)
    print(f"\nТаблица для {feat}:")
    print(cross.round(2))

# 5. Визуализация числовых признаков (boxplots)
fig, axes = plt.subplots(1, len(numeric_features), figsize=(18, 6))
for i, feat in enumerate(numeric_features):
    ax = axes[i]
    sns.boxplot(data=df, x='Exited', y=feat, ax=ax)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplots_numeric_by_churn.png"), dpi=150)
plt.close()
print("Boxplots сохранены.")

# 6. Визуализация категориальных признаков (stacked bars)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(categorical_features):
    ax = axes[i]
    # Строим сгруппированные столбцы в процентах от каждой категории
    cross = pd.crosstab(df[feat], df['Exited'], normalize='index') * 100
    cross.plot(kind='bar', stacked=True, ax=ax, color=['tab:blue', 'tab:orange'])
    ax.set_title(feat)
    ax.set_ylabel('Percentage')
    ax.legend(['Stayed', 'Churned'])
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stacked_bars_categorical_by_churn.png"), dpi=150)
plt.close()
print("Stacked bars сохранены.")

# 7. Подготовка данных для моделирования
# Кодируем категориальные признаки
df_model = df.copy()
le_geo = LabelEncoder()
le_gender = LabelEncoder()
df_model['Geography_enc'] = le_geo.fit_transform(df_model['Geography'])
df_model['Gender_enc'] = le_gender.fit_transform(df_model['Gender'])

# Выбираем признаки для модели
feature_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'Geography_enc', 'Gender_enc']

X = df_model[feature_cols]
y = df_model['Exited']

# Масштабируем числовые
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 8. Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nЛогистическая регрессия:")
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_lr):.4f}")

# Сохраняем отчёт
with open(os.path.join(output_dir, "logistic_regression_report.txt"), "w") as f:
    f.write("Logistic Regression Report\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred_lr):.4f}\n")

# 9. Случайный лес (для сравнения и важности признаков)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nСлучайный лес:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_rf):.4f}")

# Важность признаков
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nВажность признаков (Random Forest):")
print(feature_importance)
feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

# Сохраняем отчёт по случайному лесу
with open(os.path.join(output_dir, "random_forest_report.txt"), "w") as f:
    f.write("Random Forest Report\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred_rf):.4f}\n")

# 10. Визуализация важности признаков
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
plt.close()

print(f"\nВсе результаты сохранены в папку {output_dir}")

os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/model_rf.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_geo, 'models/le_geo.pkl')
joblib.dump(le_gender, 'models/le_gender.pkl')
print("Модель и предобработчики сохранены в папку 'models'.")