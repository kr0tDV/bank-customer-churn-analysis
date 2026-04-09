import joblib
import os

# Пути к сохранённым артефактам
MODEL_PATH = 'models/model_rf.pkl'
SCALER_PATH = 'models/scaler.pkl'
GEO_ENCODER_PATH = 'models/le_geo.pkl'
GENDER_ENCODER_PATH = 'models/le_gender.pkl'

# Загрузка
if not os.path.exists(MODEL_PATH):
    print("Модель не найдена. Сначала выполните scr/churn_analysis.py для обучения модели.")
    exit()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_geo = joblib.load(GEO_ENCODER_PATH)
le_gender = joblib.load(GENDER_ENCODER_PATH)

def get_user_input():
    """Сбор данных от пользователя через консоль"""
    print("\n=== Введите данные клиента ===")
    try:
        credit_score = float(input("Кредитный рейтинг (CreditScore, 350-850): "))
        age = float(input("Возраст (Age): "))
        tenure = float(input("Срок обслуживания в годах (Tenure, 0-10): "))
        balance = float(input("Баланс на счёте (Balance): "))
        salary = float(input("Предполагаемая зарплата (EstimatedSalary): "))
        num_products = float(input("Количество продуктов (NumOfProducts, 1-4): "))
        has_cr_card = int(input("Наличие кредитной карты (HasCrCard, 0 или 1): "))
        is_active = int(input("Активный участник (IsActiveMember, 0 или 1): "))
        
        geography = input("Страна (Geography, France/Germany/Spain): ").strip()
        gender = input("Пол (Gender, Male/Female): ").strip()
        
        # Кодирование
        geo_enc = le_geo.transform([geography])[0]
        gender_enc = le_gender.transform([gender])[0]
        
        features = [credit_score, age, tenure, balance, salary,
                    num_products, has_cr_card, is_active, geo_enc, gender_enc]
        return features
    except ValueError as e:
        print(f"Ошибка ввода: {e}. Попробуйте снова.")
        return None
    except Exception as e:
        print(f"Ошибка: {e}. Проверьте вводимые значения.")
        return None

def predict():
    print("="*50)
    print("СИСТЕМА ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ БАНКА")
    print("="*50)
    while True:
        user_features = get_user_input()
        if user_features is None:
            continue
        
        # Масштабирование и предсказание
        scaled = scaler.transform([user_features])
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]  # вероятность оттока
        
        print("\n=== РЕЗУЛЬТАТ ===")
        if prediction == 1:
            print(f"⚠️  Клиент СКЛОНЕН к уходу (вероятность оттока: {proba:.2%})")
            print("Рекомендация: предложить бонусы, персональные условия или связаться с клиентом.")
        else:
            print(f"✅ Клиент, вероятно, ОСТАНЕТСЯ (вероятность оттока: {proba:.2%})")
            print("Рекомендация: продолжать стандартное обслуживание.")
        
        again = input("\nПроверить другого клиента? (y/n): ").strip().lower()
        if again != 'y':
            break
    print("\nПрограмма завершена. Спасибо!")

if __name__ == "__main__":
    predict()