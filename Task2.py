import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

gender_submission = pd.read_csv('gender_submission.csv')
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

print("Info for gender_submission.csv:")
print(gender_submission.info())

print("\nInfo for test.csv:")
print(test_data.info())

print("\nInfo for train.csv:")
print(train_data.info())

print("\nDescriptive statistics for gender_submission.csv:")
print(gender_submission.describe())

print("\nDescriptive statistics for test.csv:")
print(test_data.describe())

print("\nDescriptive statistics for train.csv:")
print(train_data.describe())

print("\nMissing values in gender_submission.csv:")
print(gender_submission.isnull().sum())

print("\nMissing values in test.csv:")
print(test_data.isnull().sum())

print("\nMissing values in train.csv:")
print(train_data.isnull().sum())

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Cabin'].fillna('Unknown', inplace=True)

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
train_data['Cabin'].fillna('Unknown', inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

print("\nAfter replacing missing values with appropriate values:")
print("\nMissing values in test.csv:")
print(test_data.isnull().sum())

print("\nMissing values in train.csv:")
print(train_data.isnull().sum())

plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', data=train_data, palette='Set1')
plt.title('PassengerId vs Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=train_data, x='Pclass', palette='Set2')
plt.title('Pclass vs Total Count of Passengers')
plt.xlabel('Pclass')
plt.ylabel('Total Count')
plt.show()

age_bins = [0, 18, 35, 60, 100]
age_labels = ['0-18', '19-35', '36-60', '61+']
train_data['Age_Group'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels, right=False)

age_survived_counts = train_data.groupby('Age_Group')['Survived'].value_counts().unstack().fillna(0)
age_survived_counts.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(15, 5), cmap='viridis', legend=False)
plt.title('')
plt.suptitle('Age vs Survived', ha='center', fontsize=16)
plt.show()
