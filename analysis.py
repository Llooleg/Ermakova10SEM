# -*- coding: utf-8 -*-
"""
ПРОГНОЗИРОВАНИЕ КОЛИЧЕСТВА ИНЦИДЕНТОВ

Методы:
1. Скользящее среднее
2. Линейная регрессия
3. Полиномиальная регрессия 2-й степени
4. Экспоненциальное сглаживание
5. Пороговый метод обнаружения превышения

Файл с данными:
ермакова стат - Лист1.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# ============================================================
# НАСТРОЙКИ
# ============================================================

FILE = "total_incidents.csv"

# Размер окна скользящего среднего
WINDOW = 5

# Количество последних наблюдений для тестирования
TEST_SIZE = 20

# Степень полинома
POLY_DEGREE = 2


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

df = pd.read_csv(FILE, sep=None, engine="python")

# Преобразование даты
df["Дата"] = pd.to_datetime(df["Дата"])

# Основной показатель
y = df["Кол. инцидентов"].astype(float)

# Временной индекс
x = np.arange(len(df))

print("=" * 60)
print("АНАЛИЗ И ПРОГНОЗИРОВАНИЕ КОЛИЧЕСТВА ИНЦИДЕНТОВ")
print("=" * 60)

print(f"\nКоличество наблюдений: {len(df)}")
print("\nПервые 5 строк:")
print(df.head())

print("\nСтатистика количества инцидентов:")
print(y.describe())


# ============================================================
# 2. ФУНКЦИЯ ОЦЕНКИ КАЧЕСТВА
# ============================================================

def calculate_metrics(actual, predicted):
    """Расчёт MAE, MSE, RMSE и MAPE."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = ~np.isnan(actual) & ~np.isnan(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    # Защита от деления на ноль
    nonzero = actual != 0

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, mse, rmse, mape


# ============================================================
# 3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================================

if len(df) <= TEST_SIZE:
    raise ValueError(
        "TEST_SIZE должен быть меньше количества наблюдений."
    )

train_size = len(df) - TEST_SIZE

x_train = x[:train_size]
y_train = y.iloc[:train_size]

x_test = x[train_size:]
y_test = y.iloc[train_size:]

print("\n" + "=" * 60)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 60)

print(f"Обучающая выборка: {train_size} наблюдений")
print(f"Тестовая выборка:  {TEST_SIZE} наблюдений")


# ============================================================
# 4. СКОЛЬЗЯЩЕЕ СРЕДНЕЕ
# ============================================================

# Для каждого тестового значения используем только предыдущие
# реальные значения.

all_values = y.to_numpy()
moving_predictions = []

for i in range(train_size, len(df)):

    start = max(0, i - WINDOW)

    previous_values = all_values[start:i]

    if len(previous_values) == 0:
        prediction = np.nan
    else:
        prediction = np.mean(previous_values)

    moving_predictions.append(prediction)

moving_predictions = np.array(moving_predictions)

df["Прогноз_скользящее"] = np.nan
df.loc[train_size:, "Прогноз_скользящее"] = moving_predictions

mae_ma, mse_ma, rmse_ma, mape_ma = calculate_metrics(
    y_test,
    moving_predictions
)

print("\n" + "=" * 60)
print("1. СКОЛЬЗЯЩЕЕ СРЕДНЕЕ")
print("=" * 60)

print(f"Размер окна: {WINDOW}")
print(f"MAE :  {mae_ma:.4f}")
print(f"MSE :  {mse_ma:.4f}")
print(f"RMSE:  {rmse_ma:.4f}")
print(f"MAPE:  {mape_ma:.2f}%")


# ============================================================
# 5. ЛИНЕЙНАЯ РЕГРЕССИЯ
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    x_train.reshape(-1, 1),
    y_train
)

linear_predictions = linear_model.predict(
    x_test.reshape(-1, 1)
)

df["Прогноз_линейный"] = np.nan
df.loc[train_size:, "Прогноз_линейный"] = linear_predictions

mae_lin, mse_lin, rmse_lin, mape_lin = calculate_metrics(
    y_test,
    linear_predictions
)

print("\n" + "=" * 60)
print("2. ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 60)

print(
    f"Модель: y = {linear_model.intercept_:.4f}"
    f" + {linear_model.coef_[0]:.4f} * t"
)

print(f"MAE :  {mae_lin:.4f}")
print(f"MSE :  {mse_lin:.4f}")
print(f"RMSE:  {rmse_lin:.4f}")
print(f"MAPE:  {mape_lin:.2f}%")


# ============================================================
# 6. ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ
# ============================================================

poly = PolynomialFeatures(
    degree=POLY_DEGREE
)

x_train_poly = poly.fit_transform(
    x_train.reshape(-1, 1)
)

x_test_poly = poly.transform(
    x_test.reshape(-1, 1)
)

poly_model = LinearRegression()

poly_model.fit(
    x_train_poly,
    y_train
)

poly_predictions = poly_model.predict(
    x_test_poly
)

df["Прогноз_полином"] = np.nan
df.loc[train_size:, "Прогноз_полином"] = poly_predictions

mae_poly, mse_poly, rmse_poly, mape_poly = calculate_metrics(
    y_test,
    poly_predictions
)

print("\n" + "=" * 60)
print("3. ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ")
print("=" * 60)

print(f"Степень полинома: {POLY_DEGREE}")

print(f"MAE :  {mae_poly:.4f}")
print(f"MSE :  {mse_poly:.4f}")
print(f"RMSE:  {rmse_poly:.4f}")
print(f"MAPE:  {mape_poly:.2f}%")


# ============================================================
# 7. ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ
# ============================================================

exp_model = SimpleExpSmoothing(endog=y_train).fit(optimized=True)

exp_predictions = exp_model.forecast(
    TEST_SIZE
)

df["Прогноз_экспоненциальный"] = np.nan
df.loc[train_size:, "Прогноз_экспоненциальный"] = (
    exp_predictions.values
)

mae_exp, mse_exp, rmse_exp, mape_exp = calculate_metrics(
    y_test,
    exp_predictions
)

print("\n" + "=" * 60)
print("4. ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ")
print("=" * 60)

print(f"MAE :  {mae_exp:.4f}")
print(f"MSE :  {mse_exp:.4f}")
print(f"RMSE:  {rmse_exp:.4f}")
print(f"MAPE:  {mape_exp:.2f}%")


# ============================================================
# 8. ПОРОГОВЫЙ МЕТОД
# ============================================================

# Порог можно задать вручную.
# Например:
# threshold = 25

threshold = y.mean()

df["Порог"] = threshold

df["Превышение_порога"] = (
    y > threshold
).astype(int)

print("\n" + "=" * 60)
print("5. ПОРОГОВЫЙ МЕТОД")
print("=" * 60)

print(f"Среднее количество инцидентов: {y.mean():.2f}")
print(f"Порог: {threshold:.2f}")

print(
    "Количество наблюдений выше порога:",
    df["Превышение_порога"].sum()
)

print(
    "Доля наблюдений выше порога:",
    f"{df['Превышение_порога'].mean() * 100:.2f}%"
)


# ============================================================
# 9. СРАВНЕНИЕ МЕТОДОВ
# ============================================================

results = pd.DataFrame({
    "Метод": [
        "Скользящее среднее",
        "Линейная регрессия",
        "Полиномиальная регрессия",
        "Экспоненциальное сглаживание"
    ],

    "MAE": [
        mae_ma,
        mae_lin,
        mae_poly,
        mae_exp
    ],

    "MSE": [
        mse_ma,
        mse_lin,
        mse_poly,
        mse_exp
    ],

    "RMSE": [
        rmse_ma,
        rmse_lin,
        rmse_poly,
        rmse_exp
    ],

    "MAPE (%)": [
        mape_ma,
        mape_lin,
        mape_poly,
        mape_exp
    ]
})

results = results.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МЕТОДОВ")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# 10. ЛУЧШИЙ МЕТОД
# ============================================================

best_method = results.iloc[0]

print("\n" + "=" * 60)
print("ЛУЧШИЙ МЕТОД")
print("=" * 60)

print(f"Метод: {best_method['Метод']}")
print(f"RMSE:  {best_method['RMSE']:.4f}")
print(f"MAE:   {best_method['MAE']:.4f}")
print(f"MAPE:  {best_method['MAPE (%)']:.2f}%")


# ============================================================
# 11. ПРОГНОЗ СЛЕДУЮЩЕГО НАБЛЮДЕНИЯ
# ============================================================

next_x = np.array([[len(df)]])

# Скользящее среднее
next_ma = y.iloc[-WINDOW:].mean()

# Линейный прогноз
next_linear = linear_model.predict(
    next_x
)[0]

# Полиномиальный прогноз
next_poly = poly_model.predict(
    poly.transform(next_x)
)[0]

# Экспоненциальный прогноз
next_exp = exp_model.forecast(1).iloc[0]

print("\n" + "=" * 60)
print("ПРОГНОЗ СЛЕДУЮЩЕГО НАБЛЮДЕНИЯ")
print("=" * 60)

print(f"Скользящее среднее:          {next_ma:.2f}")
print(f"Линейная регрессия:          {next_linear:.2f}")
print(f"Полиномиальная регрессия:    {next_poly:.2f}")
print(f"Экспоненциальное сглаживание:{next_exp:.2f}")


# ============================================================
# 12. ГРАФИК ИСХОДНЫХ ДАННЫХ И ПОРОГА
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Дата"],
    y,
    label="Фактические данные"
)

plt.axhline(
    threshold,
    linestyle="--",
    label=f"Порог = {threshold:.2f}"
)

plt.xlabel("Время")
plt.ylabel("Количество инцидентов")
plt.title("Динамика количества инцидентов")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 13. ГРАФИК СРАВНЕНИЯ ПРОГНОЗОВ
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Дата"],
    y,
    label="Фактические данные"
)

plt.plot(
    df["Дата"],
    df["Прогноз_скользящее"],
    label="Скользящее среднее"
)

plt.plot(
    df["Дата"],
    df["Прогноз_линейный"],
    label="Линейная регрессия"
)

plt.plot(
    df["Дата"],
    df["Прогноз_полином"],
    label="Полиномиальная регрессия"
)

plt.plot(
    df["Дата"],
    df["Прогноз_экспоненциальный"],
    label="Экспоненциальное сглаживание"
)

plt.xlabel("Время")
plt.ylabel("Количество инцидентов")
plt.title("Сравнение методов прогнозирования")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 14. ГРАФИК ТОЛЬКО ТЕСТОВОЙ ВЫБОРКИ
# ============================================================

plt.figure(figsize=(14, 6))

test_dates = df["Дата"].iloc[train_size:]

plt.plot(
    test_dates,
    y_test,
    marker="o",
    label="Фактические значения"
)

plt.plot(
    test_dates,
    moving_predictions,
    label="Скользящее среднее"
)

plt.plot(
    test_dates,
    linear_predictions,
    label="Линейная регрессия"
)

plt.plot(
    test_dates,
    poly_predictions,
    label="Полиномиальная регрессия"
)

plt.plot(
    test_dates,
    exp_predictions,
    label="Экспоненциальное сглаживание"
)

plt.xlabel("Время")
plt.ylabel("Количество инцидентов")
plt.title("Сравнение прогнозов на тестовой выборке")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 15. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

df.to_csv(
    "результаты_прогнозирования.csv",
    index=False,
    encoding="utf-8-sig"
)

results.to_csv(
    "сравнение_методов.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)

print("Созданы файлы:")
print("1. результаты_прогнозирования.csv")
print("2. сравнение_методов.csv")
