# scr/top_active_customers.py

import pandas as pd
import os

# Пути
input_path = "data/processed/Bank_Churn_cleaned.csv"
output_dir = "result/top_active_customers"
os.makedirs(output_dir, exist_ok=True)

# Загрузка данных
df = pd.read_csv(input_path)
print(f"Загружено {len(df)} строк")

# Фильтрация: только действующие клиенты (Exited = 0)
active_df = df[df['Exited'] == 0].copy()
print(f"Действующих клиентов: {len(active_df)}")

# Сортируем по балансу (убывание), чтобы взять наиболее обеспеченных
active_sorted = active_df.sort_values(by='Balance', ascending=False)

# Теперь среди топ-N по балансу выберем самых активных и с высокой зарплатой.
# Возьмём, например, топ-50 по балансу, чтобы не упустить активных с чуть меньшим балансом, но высокой активностью.
top_balance = active_sorted.head(50).copy()

# Среди этих 50 отсортируем по: активность (убывание), зарплата (убывание), количество продуктов (убывание)
top_selected = top_balance.sort_values(
    by=['IsActiveMember', 'EstimatedSalary', 'NumOfProducts'],
    ascending=[False, False, False]
).head(10)

# Формируем результат
result_cols = ['CustomerId', 'Surname', 'Geography', 'Age', 'Balance', 'EstimatedSalary', 
               'NumOfProducts', 'IsActiveMember', 'CreditScore', 'Tenure']
top10_result = top_selected[result_cols].copy()

# Выводим в консоль
print("\nТоп-10 самых активных клиентов с наибольшим балансом и зарплатой (действующие):")
print(top10_result.to_string(index=False))

# Сохраняем в CSV
csv_path = os.path.join(output_dir, "top10_active_customers.csv")
top10_result.to_csv(csv_path, index=False)
print(f"\nТаблица сохранена: {csv_path}")
