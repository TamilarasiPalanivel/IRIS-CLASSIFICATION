import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = pd.read_csv('Irisfile.csv')

iris = iris.drop(columns=['Id'])

X = iris.iloc[:, :-1].values  
y = iris.iloc[:, -1].values   

encoder = LabelEncoder()
y = encoder.fit_transform(y)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.hist(y_pred, bins=3, alpha=0.7, color='blue', label='Predicted Species')
plt.hist(y_test, bins=3, alpha=0.5, color='green', label='Actual Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Histogram of Predicted vs Actual Species')
plt.legend()
plt.show()
