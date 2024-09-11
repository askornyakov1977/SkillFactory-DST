# %%
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from importlib import reload
import logging
reload(logging)
import funcs 
import pickle

# %%
# Задаем формат логирования
logging.basicConfig(
   format="%(levelname)s: %(asctime)s: %(message)s",
    level=logging.INFO
)

# %%
# Cоздаем лог-файл
logger = funcs.get_logger(path="logs/", file="model.log")

# %%
# Считаем файл-csv импортированный из БД мониторинга в датафрейм
file_name = 'api01.csv'
data = pd.read_csv(file_name, sep=' ')
logger.info(f"Data file: {file_name}")
logger.info(f"Data shape: {data.shape}")
logger.info(f"Data head: \n{data.head()}")

# %%
# Получим выборку с очищенными данными
data_cleaned = funcs.get_data_cleaned(data, logger=logger)

# %%
# Выделям исходные признаки и целевой признак в отдельные наборы данных
X = data_cleaned.drop('y', axis=1)
y = data_cleaned.y

# %%
# Трансформируем исходные данные 
X_transformed = funcs.get_data_transformed(X)
logger.info('Shape before transform: {}'.format(X.shape))
logger.info('Shape after transform: {}'.format(X_transformed.shape))

# %%
# Вычислим момент времени для разделения датачета на тренировочную и тестовую выборки
split_datetime = X_transformed.index.max() - pd.to_timedelta(1, 'd')
# Создадим тренировочную и тестовую выборки
X_train = X_transformed[X_transformed.index<=split_datetime]
y_train = y[y.index<=split_datetime]
X_test = X_transformed[X_transformed.index>split_datetime]
y_test = y[y.index>split_datetime]
logger.info(f'Training sample size: {X_train.shape}, begin: {X_train.index.min()}, end: {X_train.index.max()}')
logger.info(f'Test sample size: {X_test.shape}, begin: {X_test.index.min()}, end: {X_test.index.max()}')

# %%
# Выполним масштабирование исходных признаков
scaler = MinMaxScaler()
# Подгоняем параметры стандартизатора
scaler.fit(X_train)
# Производим стандартизацию тренировочной выборки
X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                              columns=X_train.columns, index=X_train.index)
# Производим стандартизацию тестовой выборки
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns, index=X_test.index)

# %%
# Объявим вспомогательную функцию по валидации данных временного ряда.
def timeseries_validation(model, X_train, y_train, n_splits=3):
  tss = TimeSeriesSplit(n_splits=n_splits, test_size=1440)
  train_test_groups = tss.split(X_train)
  metrics = []
  for train_index, test_index in train_test_groups:
    # обучаем модель
    model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    metrics.append(mean_absolute_percentage_error(y_train.iloc[test_index],
                                          model.predict(X_train.iloc[test_index])))
  return np.mean(metrics)

# %%
# Объявим целевую функцию для поиска гиперпараметров, 
# происходит инициализация модели по входящему в функцию параметру, 
# далее выполняется валидация на основе обучающей выборки, и выдается средняя метрика.
def optuna_CatBoostRegressor(trial):
            # задаем пространства поиска гиперпараметров
            model_params =  {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                #'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
                "depth": trial.suggest_int("depth", 1, 12),
                #"boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                #"bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 64),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=False),
                'random_state': trial.suggest_int("random_state", 42, 42),
                'verbose': trial.suggest_int("verbose", 0, 0)
            }
            # создаем модель
            model = CatBoostRegressor(**model_params)
            # запускаем валидацию обучающей выбоки
            score = timeseries_validation(model, X_train_scaled, y_train)
            return score

# %%
# Запустим поиск оптимальных гиперпараметров модели
sampler = TPESampler(seed=42)
optuna.logging.set_verbosity(optuna.logging.WARNING)
print('Поиск оптимальных гиперпараметров: ', 'CatBoostRegressor')
study = optuna.create_study(direction='minimize', study_name='CatBoostRegressor', sampler=sampler)
study.optimize(optuna_CatBoostRegressor, n_trials=1)
# Сохраним оптимальные гиперпараметры модели в структуру моделей
best_params = study.best_trial.params
logger.info(f'Optimal hyperparameters for the CatBoostRegressor model found: {best_params}')

# %%
# Создаём пайплайн, который включает нормализацию, отбор признаков и обучение модели
pipe = Pipeline([ 
  ('Scaling', MinMaxScaler()),
  ('CatBoostRegressor', CatBoostRegressor(**best_params))
  ])
# Обучаем пайплайн
pipe.fit(X_train, y_train);

# %%
# Объявим функцию по выводу метрик
def get_metrics(data_true, data_pred):
  metric_funcs = {r2_score: 'R2', mean_absolute_percentage_error: 'MAPE',
                  mean_absolute_error: 'MAE', mean_squared_error: 'MSE'}
  return [(metric_funcs[func], func(data_true, data_pred)) for func in metric_funcs]

# %%
# Сделаем предсказание обучающей выборки
y_pred_train = pipe.predict(X_train)
# Сохраним обучающие метрики в таблице итоговых результатов
metrics = get_metrics(y_train, y_pred_train)
logger.info(f"Learning metrics: \n{metrics}")
# Сделаем предсказание тестовой выборки
y_pred_test = pipe.predict(X_test)
# Сохраним тестовые метрики в таблице итоговых результатов
metrics = get_metrics(y_test, y_pred_test)
logger.info(f"Test metrics: \n{metrics}")

# %%
# Сериализуем pipeline и записываем результат в файл
file_name = 'pipeline.pkl'
with open(file_name, 'wb') as output:
    pickle.dump(pipe, output)
logger.info(f'The serialized pipeline is written to a file: {file_name}')

# %%
# Десериализуем pipeline из файла
with open('pipeline.pkl', 'rb') as pkl_file:
    loaded_pipe = pickle.load(pkl_file)