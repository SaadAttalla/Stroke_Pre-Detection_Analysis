import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

df = pd.DataFrame(pd.read_csv("E:/DEPI-AI&DS/Technical/Stroke project/full-stroke-data.csv"))
#df.head(5)

df['gender'].value_counts()

df = df[df['gender'].isin(['Male', 'Female'])]

#df['gender'].value_counts()

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

#df.head(5)

missing_values = df.isnull().sum()

#print("Missing values in each column:")
#print(missing_values)

#df.head(2)

df['age'] = df['age'].fillna(df['age'].mean())


Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]


age_scaler = MinMaxScaler()
df['age_scaled'] = age_scaler.fit_transform(df[['age']])

#df.head(5)

#df['age'].value_counts()

df['hypertension'] = df['hypertension'].fillna(df['hypertension'].mean())


Q1 = df['hypertension'].quantile(0.25)
Q3 = df['hypertension'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['hypertension'] >= lower_bound) &
        (df['hypertension'] <= upper_bound)]


#df.head(5)

df['heart_disease'] = df['heart_disease'].fillna(df['heart_disease'].mean())


Q1 = df['heart_disease'].quantile(0.25)
Q3 = df['heart_disease'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['heart_disease'] >= lower_bound) &
        (df['heart_disease'] <= upper_bound)]


#df.head(5)

df['ever_married'].value_counts()

df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})

#df.head(5)

df['work_type'].value_counts()

children_rows = df[df['work_type'] == 'children']

children_above_18 = children_rows[children_rows['age'] >= 18]

#print(f"Rows where work_type is 'children' and age is 18 or above:")
#print(children_above_18)

df['work_type'] = df['work_type'].map({
    'Private': 0,
    'Self-employed': 1,
    'Govt_job': 2,
    'children': 3,
    'Never_worked': 4
})

#df.head(5)

df['Residence_type'].value_counts()

df['Residence_type'] = df['Residence_type'].map({
    'Urban': 0,
    'Rural': 1,
})

#df.head(5)

df['avg_glucose_level'].value_counts()

df['avg_glucose_level'] = df['avg_glucose_level'].fillna(
    df['avg_glucose_level'].mean())


Q1 = df['avg_glucose_level'].quantile(0.25)
Q3 = df['avg_glucose_level'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['avg_glucose_level'] >= lower_bound) &
        (df['avg_glucose_level'] <= upper_bound)]

glucose_scaler = MinMaxScaler()
df['avg_glucose_level_scaled'] = glucose_scaler.fit_transform(
    df[['avg_glucose_level']])

#df.head(5)

df['smoking_status'].value_counts()

unknown_percentage = (df['smoking_status'].value_counts()[
                      'Unknown'] / len(df)) * 100
#print(f"Percentage of 'Unknown' in smoking_status: {unknown_percentage:.2f}%")

unknown_rows = df[df['smoking_status'] == 'Unknown']
#print(unknown_rows[['age', 'work_type', 'stroke']].describe())

df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
})

missing_percentage = df['bmi'].isnull().sum() / len(df) * 100
#print(f"Percentage of missing values in 'bmi': {missing_percentage:.2f}%")

bmi_missing = df[df['bmi'].isnull()]
bmi_non_missing = df[df['bmi'].notnull()]

model = RandomForestRegressor()
X_train = bmi_non_missing.drop(columns=['bmi'])
y_train = bmi_non_missing['bmi']
model.fit(X_train, y_train)


X_missing = bmi_missing.drop(columns=['bmi'])
df.loc[df['bmi'].isnull(), 'bmi'] = model.predict(X_missing)

bmi_scaler = MinMaxScaler()
df['bmi_scaled'] = bmi_scaler.fit_transform(df[['bmi']])

print(df.head(10))
