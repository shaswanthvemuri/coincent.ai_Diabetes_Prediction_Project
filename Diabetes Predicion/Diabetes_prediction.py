import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Prepare the data
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Load the dataset again for exploration
dataset = pd.read_csv("diabetes.csv")
print(dataset.head())
dataset.info()

print(dataset.isnull().sum())

print(dataset.describe())

# Correlation plot of independent variables
plt.figure(figsize =(10,8))
sns.heatmap(dataset.corr(), annot= True, fmt=".3f", cmap="YlGnBu")
plt.title("Correlation heatmap")
plt.show()

# Exploring pregnancy and target variables
plt.figure(figsize =(10,8))
# Plotting Density function graph of the pregnancies and target variables
sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1], color="Red" , shade= True)
sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0], color="Blue" , shade= True)
plt.xlabel("Pregnancies")
plt.ylabel("Density")
plt.legend(["Positive" , "Negative"])
plt.show()

# Exploring Glucose and Target variables
plt.figure(figsize=(10,8))
sns.violinplot(data=dataset, x="Outcome", y="Glucose", split=True, linewidth=2, inner="quart")
plt.show()

# Exploring Glucose and target variables
plt.figure(figsize =(10,8))
# Plotting Density function graph of the Glucose and target variables
sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1], color="Red" , shade= True)
sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0], color="Blue" , shade= True)
plt.xlabel("Glucose")
plt.ylabel("Density")
plt.legend(["Positive" , "Negative"])
plt.show()

# Replace 0 values with the mean/median of the respective feature
# Glucose
dataset["Glucose"] = dataset["Glucose"].replace(0, dataset["Glucose"].median())
# BloodPressure
dataset["BloodPressure"] = dataset["BloodPressure"].replace(0, dataset["BloodPressure"].median())
# BMI
dataset["BMI"] = dataset["BMI"].replace(0, dataset["BMI"].mean())
# SkinThickness
dataset["SkinThickness"] = dataset["SkinThickness"].replace(0, dataset["SkinThickness"].mean())
# Insulin
dataset["Insulin"] = dataset["Insulin"].replace(0, dataset["Insulin"].mean())

# Splitting the dependent and independent variable
x = dataset.drop(["Outcome"], axis=1)
y = dataset["Outcome"]

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# KNN
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
for n_neighbors in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    # Check accuracy score
    training_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.plot(range(1, 11), training_accuracy, label="training_accuracy")
plt.plot(range(1, 11), test_accuracy, label="test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)
print(knn.score(x_train, y_train), ": Training_accuracy")
print(knn.score(x_test, y_test), ": Test_accuracy")

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
print(dt.score(x_train, y_train), ": Training_accuracy")
print(dt.score(x_test, y_test), ": Test_accuracy")

dt1 = DecisionTreeClassifier(random_state=0, max_depth=3)
dt1.fit(x_train, y_train)
print(dt1.score(x_train, y_train), ": Training_accuracy")
print(dt1.score(x_test, y_test), ": Test_accuracy")

# MLP
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)
mlp.fit(x_train, y_train)
print(mlp.score(x_train, y_train), ": Training_accuracy")
print(mlp.score(x_test, y_test), ": Test_accuracy")

# Scaling the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

mlp1 = MLPClassifier(random_state=0)
mlp1.fit(x_train_scaled, y_train)
print(mlp1.score(x_train_scaled, y_train), ": Training_accuracy")
print(mlp1.score(x_test_scaled, y_test), ": Test_accuracy")
