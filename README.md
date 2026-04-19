# forest-prediction

Проект по задаче классификации типа лесного покрова на основе табличных признаков.

## Описание

В проекте реализован полный pipeline для задачи multiclass classification:
- загрузка и подготовка данных;
- feature engineering;
- раздельный preprocessing внутри pipeline;
- сравнение 5 обязательных семейств моделей через GridSearchCV;
- выбор лучшей модели по cross-validation;
- оценка на внутреннем holdout;
- сохранение лучшего полного pipeline;
- предсказание на внешнем тестовом наборе;
- построение confusion matrix и learning curve;
- отдельный EDA notebook.

## Цель

Нужно обучить модель, которая предсказывает `Cover_Type` по признакам из датасета forest cover.

Дополнительно проект должен:
- использовать holdout split внутри `train.csv`;
- использовать минимум 5-fold cross-validation;
- сравнить 5 обязательных семейств моделей;
- сохранить лучший pipeline в `pickle`;
- показать confusion matrix и learning curve;
- выдать предсказания для внешнего `test.csv`.

## Структура проекта

```text
forest-prediction/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── covtype.info
├── notebook/
│   └── EDA.ipynb
├── scripts/
│   ├── preprocessing_feature_engineering.py
│   ├── model_selection.py
│   └── predict.py
└── results/
    ├── best_model.pkl
    ├── test_predictions.csv
    └── plots/
        ├── confusion_matrix_heatmap.png
        └── learning_curve_best_model.png
```

## Что сделано

### Feature engineering

Добавлены 2 производных признака:
- `Distance_To_Hydrology` - евклидово расстояние до гидрологии на основе горизонтальной и вертикальной дистанций;
- `Fire_Road_Distance_Diff` - разность расстояний до пожарных точек и дорог.

Логика feature engineering вынесена в общий модуль и используется и при обучении, и при инференсе.

### Model selection

Сравниваются 5 обязательных семейств моделей:
- Gradient Boosting
- KNN
- Random Forest
- SVM
- Logistic Regression

Для каждой модели запускается отдельный `GridSearchCV`.

### Validation strategy

Используется следующая схема:
1. `train.csv` делится на `Train(1)` и `Test(1)` в пропорции 80/20 с `stratify=y`.
2. На `Train(1)` выполняется `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
3. По результатам CV выбирается лучшая модель.
4. Лучшая модель дополнительно оценивается на внутреннем holdout `Test(1)`.
5. Затем лучший pipeline переобучается на полном `train.csv` и сохраняется в `results/best_model.pkl`.
6. `predict.py` использует сохраненный pipeline для внешнего `data/test.csv`.

### Preprocessing

Preprocessing встроен внутрь pipeline, чтобы избежать data leakage.

Scaling применяется только к continuous-признакам и только для scale-sensitive моделей.

## Используемые файлы

### `scripts/preprocessing_feature_engineering.py`

Отвечает за:
- загрузку данных;
- разделение признаков и target;
- создание engineered features;
- сборку общего preprocessing;
- сборку model pipeline;
- создание выходных директорий.

### `scripts/model_selection.py`

Отвечает за:
- split `train.csv` на внутренние train/test;
- настройку 5 семейств моделей;
- запуск `GridSearchCV`;
- сравнение результатов;
- построение confusion matrix;
- построение learning curve;
- сохранение лучшего pipeline в `results/best_model.pkl`.

### `scripts/predict.py`

Отвечает за:
- загрузку `results/best_model.pkl`;
- чтение внешнего `data/test.csv`;
- запуск предсказаний;
- подсчет accuracy на внешнем тесте;
- сохранение `results/test_predictions.csv`.

### `notebook/EDA.ipynb`

Содержит exploratory data analysis:
- размер датасета;
- типы данных;
- распределение target;
- описательные статистики;
- анализ пропусков;
- базовые визуализации;
- краткие выводы по данным и признакам.

## Установка и запуск

### Вариант 1. `venv`

Из корня репозитория:

Windows:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запуск:

```bash
python scripts/model_selection.py
python scripts/predict.py
jupyter notebook notebook/EDA.ipynb
```

### Вариант 2. `conda`

```bash
conda env create -f environment.yml
conda activate forest-prediction
```

Запуск:

```bash
python scripts/model_selection.py
python scripts/predict.py
jupyter notebook notebook/EDA.ipynb
```

## Результаты

### Лучшая модель

По результатам cross-validation лучшей моделью стала `Random Forest`.

### Метрики

Текущие полученные метрики:
- Best CV accuracy on `Train(1)`: `0.9160`
- Holdout `Test(1)` accuracy: `0.9214`
- Train accuracy on full `train set (0)`: `0.9631`
- Final accuracy on external `test set (0)`: `0.7032`

Эти значения удовлетворяют целям проекта:
- train accuracy < `0.98`
- final external test accuracy > `0.65`

## Артефакты

После запуска создаются:
- `results/best_model.pkl` - сохраненный лучший полный pipeline;
- `results/test_predictions.csv` - предсказания для внешнего тестового набора;
- `results/plots/confusion_matrix_heatmap.png` - heatmap confusion matrix;
- `results/plots/learning_curve_best_model.png` - learning curve лучшей модели.

## Особенности
- SVM-семейство реализовано через `LinearSVC`, что соответствует SVM family.

## Краткий вывод

Проект реализует воспроизводимый pipeline выбора модели для forest cover classification с честным разделением данных, 5-fold cross-validation, feature engineering, сохранением лучшего pipeline и финальным предсказанием на внешнем тестовом наборе.

## TOC

- [Описание](#описание)
- [Цель](#цель)
- [Структура проекта](#структура-проекта)
- [Что сделано](#что-сделано)
  - [Feature engineering](#feature-engineering)
  - [Model selection](#model-selection)
  - [Validation strategy](#validation-strategy)
  - [Preprocessing](#preprocessing)
- [Используемые файлы](#используемые-файлы)
  - [`scripts/preprocessing_feature_engineering.py`](#scriptspreprocessing_feature_engineeringpy)
  - [`scripts/model_selection.py`](#scriptsmodel_selectionpy)
  - [`scripts/predict.py`](#scriptspredictpy)
  - [`notebook/EDA.ipynb`](#notebookedaipynb)
- [Установка и запуск](#установка-и-запуск)
  - [Вариант 1. `venv`](#вариант-1-venv)
  - [Вариант 2. `conda`](#вариант-2-conda)
- [Результаты](#результаты)
  - [Лучшая модель](#лучшая-модель)
  - [Метрики](#метрики)
- [Артефакты](#артефакты)
- [Особенности](#особенности)
- [Краткий вывод](#краткий-вывод)

## Авторы
- Nazar Yestayev (@nyestaye / @legion2440)
- Baktiyar Zhaksybay (@bzhaksyb)
- Dias Yelubay (@dyelubay)