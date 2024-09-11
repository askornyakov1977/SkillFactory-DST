import logging
import os.path
import pandas as pd
import datetime
from sklearn.base import TransformerMixin, BaseEstimator

# Функция для создания лог-файла и записи в него информации
def get_logger(path, file):
  """[Создает лог-файл для логирования в него]
  Аргументы:
      path {string} -- путь к директории
      file {string} -- имя файла
   Возвращает:
      [obj] -- [логер]
  """
  # проверяем, существует ли файл
  log_file = os.path.join(path, file)
 
  #если  файла нет, создаем его
  if not os.path.isfile(log_file):
      open(log_file, "w+").close()
  
  # поменяем формат логирования
  file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
  
  # конфигурируем лог-файл
  logging.basicConfig(level=logging.INFO, 
  format = file_logging_format)
  logger = logging.getLogger()
  
  # создадим хэнлдер для записи лога в файл
  handler = logging.FileHandler(log_file)
  
  # установим уровень логирования
  handler.setLevel(logging.INFO)
  
  # создадим формат логирования, используя file_logging_format
  formatter = logging.Formatter(file_logging_format)
  handler.setFormatter(formatter)
  
  # добавим хэндлер лог-файлу
  logger.addHandler(handler)
  return logger

  # Объявим функцию по заполнению пропусков и выбросов сквозной медианой
def get_data_cleaned(data, *, logger, confidence_level=0.70):
    data = data.copy()
    # Создадим отдельное поле по часам и минутам без секунд
    data['time_short'] = data['time'].apply(lambda x: x[:5])
    # Зададим cкозной интервал квантилей
    boundaries = round((1 - confidence_level) / 2, 2)
    quantiles = [boundaries, confidence_level + boundaries]
    # Создадим сгруппированный датасет агрегирующий значения временного ряда по
    # квантилям нижнего и верхнего уровня доверительного интервала и медиане
    data_groupby_time = data.groupby(['time_short']).value.agg([('lower_bound',
                                                        lambda x: x.quantile(quantiles[0])), 'median',
                                                        ('upper_bound', lambda x: x.quantile(quantiles[1]))])
    # На случай пропуска каких либо минут заполним их методом интерполяции
    logger.info(f'Number of entries before filling in gaps: {data_groupby_time.shape[0]}')
    data_groupby_time.index = pd.to_datetime(data_groupby_time.index, format='%H:%M')
    data_groupby_time = data_groupby_time.resample('min').mean().interpolate('linear')
    logger.info(f'Number of entries after filling in the blanks: {data_groupby_time.shape[0]}')
    logger.info(f'Basic statistical characteristics of end-to-end values of a time series: \n {data_groupby_time.describe()}')

    # Сгенирируем новый признак включающий дату и время, без учета секунд.
    data['date_time'] = pd.to_datetime(
            data[['date', 'time_short']].apply(lambda x: x.iloc[0]+' '+x.iloc[1], axis=1),
            format='%d.%m.%Y %H:%M')
    # Для удобства работы с данными, новый признак сделаем индексом. Удалим лишние поля.
    data = data.set_index('date_time').sort_index(ascending=True)
    data = data.drop(['date', 'time', 'time_short', 'id'], axis=1)
    data.columns = ['y']
    logger.info(f'Transformed data: \n{data.head()}')
    
    logger.info(f'Number of records before filling gaps and merging duplicates: {data.shape[0]}')
    # Добавим недостающие минуты с пустыми значениями
    data = data.resample('min').mean()
    logger.info(f'Number of records after filling gaps and merging duplicates: {data.shape[0]}')

    # Создадим функцию возвращающую сквозное медианное значение,
    # соответствующее часу и минуте, к которым относится входящее значение,
    # если входящее значение не находится в пределах доверительного интервала,
    # иначе функция возвращает входящее значение
    def get_norm_y(series):
        y = data_groupby_time.loc[datetime.time(series.name.hour, series.name.minute)].iloc[0]
        if  series.y > y.lower_bound and series.y < y.upper_bound:
            return series.y
        else:
            return y['median']
    # Создадим отдельный целевой признак очищенный от выбросов и пустых значений,
    # путем их замены скозным медианным значением
    data['y_clean'] = data.apply(get_norm_y, axis=1)
    logger.info(f'Data with a new target feature: \n {data.describe()}')

    # Удалим неочищенный целевой признак, и переименуем очищенный в убодный вид
    data = data.drop('y', axis=1).rename(columns={'y_clean': 'y'})
    return data

# Объявим функцию генерации временный прихнаков  
def get_data_transformed(X):
    X = X.copy()
    # Сгененируем временные признаки
    X['age_in_day'] = (X.index.max() - X.index).days
    X['hour'] = X.index.hour
    X['minute'] = X.index.minute
    X = pd.get_dummies(X, columns=['hour', 'minute'])
    return X