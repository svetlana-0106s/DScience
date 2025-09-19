import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#1. Загрузка данных
def load_data(file_path):
  return pd.read_csv('student_performance.csv')
  
df = pd.read_csv('student_performance.csv')

#2. Информация о данных
print(df.head())
print(df.info())
print(df.describe())

#student_id → Уникальный идентификатор для каждого ученика.
#weekly_self_study_hours → Среднее количество часов самообучения в неделю.
#attendance_percentage → Процент посещаемости.
#class_participation → балл от 0 до 10, указывающий на то, насколько активно учащийся участвует в занятиях.
#total_score → Итоговая оценка успеваемости (0–100).
#grade → Категориальный знак (A, B, C, D, F), производный от total_score.

#3. Обработка данных
#Проверяем наличие пропущенных значений
print('Наличие пропущенных значений:')
print(df.isnull().sum())

#Проверяем наличие дубликатов
print('\nНаличие дубликатов:', df.duplicated().sum())

#Удаляем столбец с id
df= df.drop('student_id', axis=1) 

#4. Визуализация данных
#Гистограмма
sns.set(style="whitegrid")
df.hist(bins=20, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Общий балл')
plt.ylabel('Частота')
plt.title('Распределение общего балла')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.countplot(x='grade', data=df)
plt.title('Распределение баллов')
plt.xlabel('Класс')
plt.ylabel('Значение')
plt.show()


def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Определение числовых и категориальных признаков
    numeric_features = ['weekly_self_study_hours', 'attendance_percentage', 'class_participation', 'total_score']
    categorical_features = ['grade']

    # Создание препроцессора
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # Применение препроцессора к данным
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2