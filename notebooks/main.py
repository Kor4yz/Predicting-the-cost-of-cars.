# Весь файл запускать сразу не следует!

# Подключаем библиотеки и загружаем базу данных
!pip install category_encoders
import numpy as np
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
from tensorflow import keras
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

url = 'https://drive.google.com/file/'
url = 'https://drive.google.com' + url.split('/')[-2]
df = pd.read_csv(url)
df.head()

# Подготавливаем данные
df.info()

# проверим количество нулевых элементов в нашем датасете для каждого из параметров
df.isnull().sum()

# выбросим все строки содержащие нулевые элементы
df.dropna(inplace=True)
# как видим все они удалились
df.isnull().sum()

# Теперь рассмотрим три параметра seller, VIN, saledate vs. sallingprice

# посмотрим кол-во продавцов
df['seller'].nunique()

# тут захэшируем наших продавцов
sellers = []
seller = list(df['seller'])
for el in seller:
    sellers.append((abs(hash(el)) % (10**8)) / 10000000)

pd.DataFrame(sellers).nunique() # уникальных значений осталось столько же

plt.figure(figsize=(15, 5))
plt.scatter(sellers, df['sellingprice'], color='red')
plt.xlabel("seller")
plt.ylabel('Selling Price')
plt.title("Linear relation between seller and selling price")
plt.tight_layout()
plt.show()

# Во-первых продавцов много, во-вторых от продовца стоимость не зависит, по-этому исключим это.

# так теперь обработаем дату продажи, если она в неправильном формате, то сопаставим ноль, иначе количество дней от начала минимального года
def class_month(a):
    if a == 'Jan':
        return 0
    if a == 'Feb':
        return 31
    if a == 'Mar':
        return 59
    if a == 'Apr':
        return 90
    if a == 'May':
        return 120
    if a == 'Jun':
        return 151
    if a == 'Jul':
        return 181
    if a == 'Aug':
        return 212
    if a == 'Sep':
        return 243
    if a == 'Oct':
        return 273
    if a == 'Nov':
        return 304
    if a == 'Dec':
        return 334

sale_date = df['saledate']
n_sale_date = []
for el in sale_date:
    n_sale_date.append(el.split())

# пример одной правильно заполненной ячейки
n_sale_date[1]

years = []
for el in n_sale_date:
    years.append(int(el[3]))

min_year = min(years)
f_sale_date = []
for el in n_sale_date:
    if len(el) == 7:
        f_sale_date.append(int(el[2]) + class_month(el[1]) + ((int(el[3]) - min_year) * 365))
    else:
        f_sale_date.append(0)

plt.figure(figsize=(15, 5))
plt.scatter(f_sale_date, df['sellingprice'], color='green')
plt.xlabel("saledate")
plt.ylabel('Selling Price')
plt.title("Linear relation between saledate and selling price")
plt.tight_layout()
plt.show()

# Вообщем получаем, что цена никак особо и не зависит от продажи, у нас так же есть нулевые элементы, а это значит, что в некоторых записях дата в некоректном виде. В середине есть "просадка", но там просто меньше машин было продано, поэтому и разнообразность цен меньше.

# так как база данных слишком большая возьмем только ее часть
df = df.sample(frac=0.6, random_state=1, ignore_index=True)
# так же выкинем столбцы с вин, так как он у всех машин разный,
# продавцом и датой продажи, так как они не влияют
df.drop(['seller', 'vin', 'saledate'], axis=1, inplace=True)
print(df.shape)
df.head()

# определим функцию которая будет нам выводить количество уникальных данных для каждой категории нашего датасета
categ = ['make', 'model', 'trim', 'body', 'state', 'color', 'interior', 'transmission']
def printUniqueAmount(df):
    for col in categ:
        print(f"{col}: {df[col].nunique()}")

printUniqueAmount(df)

# если какая-то категория встречается меньше 10 раз, то мы заменяем на 'Other'
for col in categ:
    amount = df[col].value_counts()
    rare_categories = amount[amount <= 10].index.tolist()
    df[col] = df[col].apply(lambda x: 'Other' if x in rare_categories else x)

# теперь наше кол-во уникальных категорий:
printUniqueAmount(df)

# год выпуска машины меняем на возраст машины
df['cars_age'] = df['year'].apply(lambda x: 2015 - x)
# transmissiom переводим automatic - 0, manual - 1
df = pd.get_dummies(data=df, columns=['transmission'], drop_first=True, dtype=int)
# так же переводим наши категориальные данные с помощью BinaryEncoder
encoder = ce.binary.BinaryEncoder(cols=['make', 'body', 'interior', 'color', 'state'], drop_invariant=True).fit(df)
df = encoder.transform(df)
# так же преобразуем с помощью hasher текстовые данные
hasher = FeatureHasher(n_features=40, input_type='string')
hashed_features = hasher.transform(df[['model', 'trim']].astype(str).to_numpy())
hashed_features_df = pd.DataFrame(hashed_features.toarray())
hashed_features_df.columns = ['feature_' + str(i) for i in range(hashed_features_df.shape[1])]
# убираем year, model, trim, потому что мы их заменили другими
df.drop(['year', 'model', 'trim'], axis=1, inplace=True)
# склеим все получившиеся данные
df = pd.concat([df, hashed_features_df], axis=1)
df.head()

df.info()

# предсказывать будем стоимость машины
# разделяем на обучающий и тестовый наборы
X = df.drop(['sellingprice'], axis=1)
y = df['sellingprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# нормализуем наши данные для лучших результатов обучения
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# посмотрим на наши поолучившиеся значения после нормализации
print("Набор для обучения X:\n", X_train)
print("Тестовый набор X:\n", X_test)


# Функция для оценки результатов
# функция для оценки результатов
def print_res(pred, y_test, pred_train, y_train):
    delta = pred_train - y_train
    absDelta = abs(delta)
    print("Средняя ошибка для обучающего набора: ")
    print(sum(absDelta) / len(absDelta))
    delta = pred - y_test
    absDelta = abs(delta)
    print("Средняя ошибка для тестового: ")
    print(sum(absDelta) / len(absDelta))
    plt.scatter(y_test, pred, label='test')
    plt.scatter(y_train, pred_train, label='train')
    plt.xlabel('Правильные значение')
    plt.ylabel('Предсказания')
    plt.legend()
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.show()
    print("Для обучающего набора:")
    print('Реальное значение:\\n', y_train[:5], "\nПредсказанное значение:\n", pred_train[:5])
    print("Для тестовых:")
    print('Реальное значение\n', y_test[:5], "\nПредсказанное значение\n", pred[:5])


# Создаем и обучаем нейросеть
# создаем модель нейронки
model = keras.Sequential([
    keras.Input(shape=(73,)),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# теперь посмотрим нашу модель на тестовых данных и ошибки которые она дает
test_loss = model.evaluate(X_test, y_test)
test_loss

# построим графики ошибки от эпох обучения
plt.plot(history.history['mae'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# посмотрим какая средняя разница между предугаданными значениями и реальными
pred = model.predict(X_test)
pred = pred.flatten()
pred_train = model.predict(X_train)
pred_train = pred_train.flatten()
print_res(pred, y_test, pred_train, y_train)

# После ста эпох ошибка на проверочном наборе начинает возрастать, поэтому выберем 100 эпох
# Получаем хорошие результаты.
# Теперь обучим нормализованным данным.

# Нормализация данных
# нормализуем данные y_train
yScaler = StandardScaler()
yScaler.fit(np.array(y_train).reshape(-1, 1))
# нормализуем по нормальному распределению
yTrainScaled = yScaler.transform(np.array(y_train).reshape(-1, 1))
print(yTrainScaled.shape)
print(y_train[1])
print(yTrainScaled[1])

# нормализуем данные y_test
yScalerTest = StandardScaler()
yScalerTest.fit(np.array(y_test).reshape(-1, 1))
# нормализуем по нормальному распределению
yTestScaled = yScalerTest.transform(np.array(y_test).reshape(-1, 1))
print(yTestScaled.shape)
print(y_train[1])
print(yTestScaled[1])

# посмотрим, что у нас получилось
print("Train:")
print(min(y_train), max(y_train))
print(min(yTrainScaled), max(yTrainScaled))
print("Test:")
print(min(y_test), max(y_test))
print(min(yTestScaled), max(yTestScaled))


# Нейросеть на нормализованных данных
# создаем модель нейронки
modelS = keras.Sequential([
    keras.Input(shape=(73,)),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1),
])
modelS.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
historyS = modelS.fit(X_train, yTrainScaled, epochs=100, validation_split=0.2)

# теперь посмотрим нашу модель на тестовых данных и ошибки которые она дает
test_loss_S = modelS.evaluate(X_test, yTestScaled)
test_loss_S

# построим графики ошибки от эпох обучения
plt.plot(historyS.history['mae'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(historyS.history['val_mae'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# делаем предсказание, приводим его к начальному виду и находим среднее значение ошибки
pred = modelS.predict(X_test)
predUnscaled = yScaler.inverse_transform(pred).flatten()
pred_train = modelS.predict(X_train)
predUnscaled_train = yScaler.inverse_transform(pred_train).flatten()
print_res(predUnscaled, y_test, predUnscaled_train, y_train)

# При увелечении количества эпох ошибка для теста увеличивается

# AutoML
!pip install autokeras
'''!!!тут надо будет согласится с перезапуском, импортировать autokeras, а затем запусть все клетки, кроме секции (Создаем и обучаем нейросеть,
нейросеть на нормализованных данных)'''

# будем использовать AutoKeras
import autokeras as ak

# Обучим на нормализованных данных
# тут создадим модель AutoKeras
modelAS = ak.AutoModel(
    inputs=[ak.Input()],
    outputs=[ak.RegressionHead()],
    max_trials=5
)
history_AS = modelAS.fit(
    [X_train],
    [yTrainScaled],
    epochs=150
)

# теперь посмотрим нашу модель на тестовых данных и ошибки которые она дает
test_loss = modelAS.evaluate(np.array(X_test), np.array(yTestScaled))
test_loss

# построим графики ошибки от эпох обучения
plt.plot(history_AS.history['mean_squared_error'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# делаем предсказание, приводим его к начальному виду и находим среднее значение ошибки
pred = modelAS.predict(X_test)
predUnscaled = yScaler.inverse_transform(pred).flatten()
pred_train = modelAS.predict(X_train)
predUnscaled_train = yScaler.inverse_transform(pred_train).flatten()
print_res(predUnscaled, y_test, predUnscaled_train, y_train)

# Теперь сделаем модель на ненормированных данных y_train и y_test.
# тут создадим модель AutoKeras
modelA = ak.AutoModel(
    inputs=[ak.Input()],
    outputs=[ak.RegressionHead()],
    max_trials=3
)
history_A = modelA.fit(
    [X_train],
    [np.array(y_train)]
)

# теперь посмотрим нашу модель на тестовых данных и ошибки которые она дает
test_loss = modelA.evaluate(np.array(X_test), np.array(y_test))
test_loss

# делаем предсказание
pred = modelA.predict(X_test)
pred_train = modelA.predict(X_train)

# построим графики ошибки от эпох обучения
plt.plot(history_A.history['mean_squared_error'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

print_res(pred.reshape(1, -1).squeeze(), np.array(y_test), pred_train.reshape(1, -1).squeeze(), np.array(y_train))

# Получили очень хорошие результаты
