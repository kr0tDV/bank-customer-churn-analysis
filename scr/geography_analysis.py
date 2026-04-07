import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Настройки
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Пути
processed_data_path = "data/processed/Bank_Churn_cleaned.csv"
output_dir = "result/geography_analysis"
os.makedirs(output_dir, exist_ok=True)

# 1. Загрузка данных
df = pd.read_csv(processed_data_path)
print(f"Загружено {len(df)} строк, столбцы: {list(df.columns)}")

# 2. Общее количество клиентов по регионам
region_counts = df['Geography'].value_counts().reset_index()
region_counts.columns = ['Geography', 'Count']
print("\nКоличество клиентов по регионам:")
print(region_counts)

# Сохраняем таблицу
region_counts.to_csv(os.path.join(output_dir, "customer_counts_by_region.csv"), index=False)

# 3. Подготовка признаков для визуализации
# Выбираем интересующие признаки (можно добавить/убрать по желанию)
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
categorical_features = ['Gender', 'HasCrCard', 'IsActiveMember', 'Exited', 'NumOfProducts']

# Для числовых признаков будем строить гистограммы с плотностью
# Для категориальных – столбчатые диаграммы (доли)

# Создадим один большой рисунок: строки – признаки, столбцы – страны
# Всего строк = len(numeric_features) + len(categorical_features)
n_rows = len(numeric_features) + len(categorical_features)
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
fig.suptitle("Сравнение распределений признаков по странам", fontsize=20, y=1.02)

# Список стран и соответствующие им данные
countries = ['Germany', 'France', 'Spain']
colors = {'Germany': 'tab:blue', 'France': 'tab:orange', 'Spain': 'tab:green'}

# Для каждого числового признака строим гистограммы
for i, feature in enumerate(numeric_features):
    for j, country in enumerate(countries):
        ax = axes[i, j]
        data = df[df['Geography'] == country][feature].dropna()
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, color=colors[country], alpha=0.6, bins=30)
            ax.set_title(f"{country}\n{feature}")
            ax.set_xlabel(feature)
        else:
            ax.text(0.5, 0.5, f"No data for {country}", ha='center', transform=ax.transAxes)
            ax.set_title(f"{country}\n{feature}")

# Для каждого категориального признака строим столбчатые диаграммы (доли)
offset = len(numeric_features)
for i, feature in enumerate(categorical_features):
    for j, country in enumerate(countries):
        ax = axes[offset + i, j]
        data = df[df['Geography'] == country][feature].value_counts(normalize=True).reset_index()
        data.columns = [feature, 'proportion']
        # Сортировка для красоты (если категории нечисловые)
        if feature in ['Gender', 'HasCrCard', 'IsActiveMember', 'Exited']:
            # Для бинарных признаков лучше оставить порядок 0,1 или Female, Male
            data = data.sort_values(feature)
        sns.barplot(data=data, x=feature, y='proportion', ax=ax, palette='viridis')
        ax.set_title(f"{country}\n{feature}")
        ax.set_ylabel("Доля")
        # Поворот подписей, если нужно
        ax.tick_params(axis='x', rotation=45)

# Подгоняем layout
plt.tight_layout()
plt.subplots_adjust(top=0.95)
# Сохраняем весь рисунок
output_fig = os.path.join(output_dir, "feature_distributions_by_country.png")
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
plt.close()

print(f"График сохранён: {output_fig}")

# 4. Дополнительно: числовые сводки по регионам
print("\nСводные статистики по регионам:")

summary_stats = []
for feature in numeric_features:
    for country in countries:
        data = df[df['Geography'] == country][feature]
        if len(data) > 0:
            stats = {
                'Feature': feature,
                'Geography': country,
                'Mean': data.mean(),
                'Median': data.median(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max()
            }
            summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
print(summary_df.round(2))

# Сохраняем сводку
summary_df.to_csv(os.path.join(output_dir, "numeric_summary_by_region.csv"), index=False)

# Для категориальных признаков можно сохранить таблицы частот
for feature in categorical_features:
    freq = pd.crosstab(df['Geography'], df[feature], normalize='index') * 100
    freq.round(2).to_csv(os.path.join(output_dir, f"crosstab_{feature}_by_region.csv"))
    print(f"\nТаблица для {feature} сохранена.")

print("\nАнализ завершён. Все результаты в папке:", output_dir)