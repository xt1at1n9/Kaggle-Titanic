import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

pd.options.mode.chained_assignment = None

train_set_raw = "/Users/uttaran/Desktop/train.csv"
df0 = pd.read_csv(train_set_raw, header = 0)
train_set = df0[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Survived']]

test_set_raw = "/Users/uttaran/Desktop/test.csv"
df1 = pd.read_csv(test_set_raw, header = 0)
test_set = df1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

def clean(df):
    df['Age'].fillna(-1, inplace=True)
    df.fillna(0, inplace=True)

    for i in range(len(df)):
        if df['Cabin'][i] == 0:
            df['Cabin'][i] = 0
        else:
            df['Cabin'][i] = 1

    d = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(d)

    d = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(d)

    return df


clean(train_set)
clean(test_set)

train_set['Embarked'][61] = -1
train_set['Embarked'][829] = -1

# print(test_set['SibSp'].isnull().values.any())
# index = test_set['Age'].index[test_set['Age'].apply(np.isnan)]
# print(index)


# print(test_set['Embarked'][61])
# print(test_set['Embarked'][11])

# print(test_set[50:100])


features = list(train_set.columns[:8])

X = train_set[features]
y = train_set['Survived']


clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)


prediction = clf.predict(test_set[features])

with open('/Users/uttaran/Desktop/solutions0.csv', 'w') as new_file:
    csv_writer = csv.writer(new_file)
    xi = "PassengerId" , "Survived"
    csv_writer.writerow(xi)
    for i in range(len(df1)):
        xi = df1['PassengerId'][i] , prediction[i]
        csv_writer.writerow(xi)
