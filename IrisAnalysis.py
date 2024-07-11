import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/DaniCodeHub/data/master/iris.csv')
print(df)

y = df['Species']
print(y)

X = df.drop('Species', axis=1)
print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100, shuffle=True)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=4, ccp_alpha=0.01)
dtc = dtc.fit(X_train, y_train)

predictions = dtc.predict(X_test)
print(predictions)

# Label Classes Explicitly Using Numpy #
all_classes = np.unique(y)

from sklearn.metrics import accuracy_score, f1_score

accuracy_score(y_test, predictions)
f1_score(y_test, predictions, labels=all_classes, average='weighted', zero_division=0)

print(f"Accuracy Score: {accuracy_score(y_test, predictions)}")

print(f'F1 Score: {f1_score(y_test, predictions, labels=all_classes, average='weighted', zero_division=0)}')

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn import tree
feature_names = X.columns
print(feature_names)

feature_importance = pd.DataFrame(dtc.feature_importances_, index=feature_names).sort_values(0, ascending=False)
print(feature_importance)

features = list(feature_importance[feature_importance[0]>0].index)
print(features)

feature_importance.head(10).plot(kind='bar')

plt.xlabel('Species')
plt.ylabel('Importance')
plt.title('Feature_importance')
plt.show()

fig = plt.figure(figsize=(10, 10))
_ = tree.plot_tree(dtc,
                   feature_names=feature_names,
                   class_names={0: 'setosa', 1: 'versicolor', 2: 'virginica'},
                   filled=True,
                   fontsize=12)
plt.show()
