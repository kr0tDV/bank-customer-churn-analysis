import os
import pandas as pd 

# Пути (учитываем структуру: папка data/raw и data/processed)
raw_dir = "data/raw"
processed_dir = "data/processed"
raw_file = os.path.join(raw_dir, "Bank_Churn.csv")
processed_file = os.path.join(processed_dir, "Bank_Churn_cleaned.csv")

# Создаём папку для очищенных данных, если её нет
os.makedirs(processed_dir, exist_ok=True)

# 1. Размер исходного файла
if os.path.exists(raw_file):
    size_bytes = os.path.getsize(raw_file)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Исходный файл: {raw_file}")
    print(f"Размер: {size_bytes} байт ({size_mb:.2f} МБ)")
else:
    print(f"Файл {raw_file} не найден!")
    exit()

# 2. Загрузка данных
df = pd.read_csv(raw_file)
print(f"\nИсходные данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 3. Удаление строк с пропусками
df_clean = df.dropna()
print(f"После удаления пропусков: {df_clean.shape[0]} строк, {df_clean.shape[1]} столбцов")

# 4. Удаление дубликатов по первичному ключу 
primary_key = "CustomerId"
if primary_key in df_clean.columns:
    # Сохраняем количество строк до удаления дубликатов
    before_dedup = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates(subset=[primary_key], keep='first')
    print(f"После удаления дубликатов по {primary_key}: {df_clean.shape[0]} строк (удалено {before_dedup - df_clean.shape[0]})")
else:
    print(f"Столбец {primary_key} не найден. Проверка дубликатов по всем столбцам.")
    before_dedup = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    print(f"После удаления полных дубликатов: {df_clean.shape[0]} строк (удалено {before_dedup - df_clean.shape[0]})")

# 5. Информация о колонках (типы, количество непустых, процент заполненности)
print("\nИнформация о колонках (очищенные данные):")
print("-" * 60)
info = pd.DataFrame({
    'Column': df_clean.columns,
    'Type': df_clean.dtypes.values,
    'Non-Null Count': df_clean.count().values,
    'Null Count': df_clean.isnull().sum().values,
    'Null %': (df_clean.isnull().sum() / len(df_clean) * 100).values
})
print(info.to_string(index=False))

# 6. Первые 5 строк очищенных данных
print("\nПервые 5 строк очищенных данных:")
print(df_clean.head())

# 7. Сохранение очищенных данных
df_clean.to_csv(processed_file, index=False)
print(f"\nОчищенные данные сохранены в: {processed_file}")

# 8. Размер очищенного файла
if os.path.exists(processed_file):
    clean_size_bytes = os.path.getsize(processed_file)
    clean_size_mb = clean_size_bytes / (1024 * 1024)
    print(f"Размер очищенного файла: {clean_size_bytes} байт ({clean_size_mb:.2f} МБ)")