# Предсказание стоимости автомобилей 🚗💸

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Kor4yz/Predicting-the-cost-of-cars./ci.yml?label=CI)](../../actions)
[![Made with: Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#стек)

Модель регрессии для предсказания рыночной цены авто на табличных данных: EDA → фичи → отбор/скейлинг → ансамбли (GBM) → сравнение метрик.

## Задача
- Очистить и исследовать датасет автомобилей (марка, модель, год, пробег, мощность/объём, тип топлива/трансмиссии и др.).
- Построить несколько моделей (Linear/Ridge/Lasso, RandomForest, GradientBoosting, XGBoost/LightGBM), подобрать гиперпараметры.
- Оценить качество (*MAE, RMSE, R²*), интерпретировать факторы цены (важности признаков, PDP/ICE, SHAP).

## Данные
- `data/raw/` — исходные CSV (не в репозитории; см. ниже DVC/скачивание).
- `data/processed/` — после очистки/фичеинжиниринга.
- Признаки: `brand`, `model`, `year`, `mileage`, `engine_power`, `engine_volume`, `fuel`, `transmission`, `body`, `owners`, `region`, пр.

> Пример графиков см. в [`Screenshots/`](Screenshots).

## Результаты (пример)
| Модель            | MAE        | RMSE       | R²    |
|-------------------|-----------:|-----------:|:-----:|
| RidgeCV           | **XXXXX**  | YYYYY      | 0.ZZ  |
| RandomForest      | XXXXX      | YYYYY      | 0.ZZ  |
| XGBoost           | XXXXX      | **YYYYY**  | **0.ZZ** |

📊 **Важности признаков / SHAP** — в отчёте и в папке [`Screenshots/`](Screenshots).

## Быстрый старт
```bash
git clone https://github.com/Kor4yz/Predicting-the-cost-of-cars..git
cd Predicting-the-cost-of-cars.
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make train           # полный цикл: load → preprocess → train → evaluate
make report          # сгенерировать отчёт с метриками/графиками
```
Стек

Python (pandas, numpy, scikit-learn, xgboost/lightgbm), Jupyter, matplotlib/seaborn, DVC (опц.), pre-commit, GitHub Actions.

Воспроизводимость

Фиксация версий в requirements.txt.

Сид генератора случайных чисел.

Разделение train/valid по стратифицированному фолдингу (по году/бренду).

(Опционально) DVC для версионирования data/.

Отчёты и материалы

📄 Report.pdf — краткий отчёт по проекту.

🎞 Presentation.pptx — презентация с результатами.

🖼️ Папка Screenshots/
 — ключевые графики (распределения, важности, ошибки).
## 📬 Автор
**Денис Морозов**  
📧 Kor4yz@yandex.ru · [GitHub](https://github.com/Kor4yz) · [Telegram](https://t.me/kor4yz)
