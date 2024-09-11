# %%
import pandas as pd
import numpy as np
import datetime
from scipy.stats import t
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
logger = funcs.get_logger(path="logs/", file="forecast.log")

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
# Производим десериализацию и извлекаем модель из файла формата pkl
with open('pipeline.pkl', 'rb') as pkl_file:
    pipe = pickle.load(pkl_file)

# %%
index_test = pd.DatetimeIndex(pd.date_range(data_cleaned.index.max() + pd.Timedelta(1, 'min'), periods=1440, freq='min'),
                name='date_time')
X = pd.DataFrame(index=index_test)

# %%
# Трансформируем исходные данные 
X_transformed = funcs.get_data_transformed(X)
logger.info('Shape before transform: {}'.format(X.shape))
logger.info('Shape after transform: {}'.format(X_transformed.shape))

# %%
# Сделаем предсказание обучающей выборки
y_pred = pipe.predict(X_transformed)

# %%
# Объявим вспомогательную функцию по вычислению остаточной стандартной ошибки: residual standard error
def get_rse(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values
    """
    y_true = np.array(y_true)
    n = y_true.shape[0]
    if n - 2 > 0:
        y_predicted = np.array(y_predicted)
        rss = np.sum(np.square(y_true - y_predicted))
        return np.sqrt(rss / (n - 2))
    else:
        return 0

# %%
# Зададим параметры вычисления 
prediction_level = 0.95
boundaries = round((1 - prediction_level) / 2, 2)
quantiles = [boundaries, prediction_level + boundaries]
# Зададим степень уверенности
gamma = prediction_level
# Уровень значимости
alpha = 1 - gamma
# Зададим продолжительность сезона в минутах
season = 1440
# Размер выборки
n = int(data_cleaned.shape[0] / season)
# Число степеней свободы
k = n - 1
# t-критическое
t_crit = -t.ppf(alpha/2, k)
# Объявим функцию для вычисления границы интервала прогнозирования для каждой минуты тестовой выборки
def get_prediction_interval(series):
  # Получим серию значений временного ряда за аналогичные минуты в пред. n суток
  y = data_cleaned.y[data_cleaned.index.time == datetime.time(series.name.hour, series.name.minute)]
  # Расчитаем отдельно верхную и нижнюю границы
  # Отфильтруем актуальные значения отдельно
  y_true_upper = y[y>series.y]
  y_true_lower = y[y<series.y]
  # Получим остаточную стандартную ошибку
  rse_upper = get_rse(y_true_upper, series.y)
  rse_lower = get_rse(y_true_lower, series.y)
  # погрешность
  eps_upper = t_crit * rse_upper
  eps_lower = t_crit * rse_lower
  # левая (нижняя) граница
  lower_bound = series.y - eps_lower
  # правая (верхняя) граница
  upper_bound = series.y + eps_upper
  return lower_bound, upper_bound

# %%
# Вычислим границы интервала прогнозирования для каждой минуты прогнозной выборки
# Cохраним результат в выборку для хранения прогноза
df_y_pred = pd.DataFrame({'y': y_pred}, index=X_transformed.index)
df_y_pred[['lower_bound', 'upper_bound']] = df_y_pred.apply(get_prediction_interval, axis=1).to_list()

# %%
logger.info(f'Main statistical characteristics of the model predictions: \n{df_y_pred.describe()}')

# %%
# Сохраним предсказания на сутки вперед в csv файл.
file_name = 'prediction.csv' 
df_y_pred.to_csv(file_name)
logger.info(f'Model predictions are saved to file: {file_name}')