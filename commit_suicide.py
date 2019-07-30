import pandas as pd
import math

def convert_income_to_numeric(income):
    income_in_numeric = []
    for i in income :
        temp = str(i).replace('$', '')
        temp = temp.replace('to', '')
        temp = temp.replace(',', '')
        currency = temp.split(' ')
        income_in_numeric.append(int(currency[0]))
    return income_in_numeric
    
dataset = pd.read_csv('foreveralone.csv')
X = dataset.iloc[:, [2, 4, 6, 7, 10, 11, 12, 17]].values
X_temp = dataset.iloc[:, [2, 4, 6, 7, 10, 11, 12, 17]]
X_dataframe = pd.DataFrame(X)

X[:, 1] = convert_income_to_numeric(X[:, 1])
Y = dataset.iloc[:, 14].values
#x = pd.DataFrame(X)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
Y = labelencoder.fit_transform(Y)

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

onehotencoder = OneHotEncoder(categorical_features=[6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]

onehotencoder = OneHotEncoder(categorical_features=[8])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]

onehotencoder = OneHotEncoder(categorical_features=[9])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]]

onehotencoder = OneHotEncoder(categorical_features=[10])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, prediction).tolist()
model_accuracy = math.ceil(((conf_matrix[0][0] + conf_matrix[1][1]) * 100)/ len(y_test))

"""
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, prediction)
""






