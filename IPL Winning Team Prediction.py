import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
data = pd.read_csv('ipl match data.csv')
#print the first 10 rows of the dataset
print(data.head(10))

# Check for Null values
print("Null values in each column:")
print(data.isnull().sum())

# Drop rows with Null values in 'winner'
data.dropna(subset=['winner'], inplace=True)

# Select features and target variable
X = pd.get_dummies(data[['team1', 'team2', 'venue', 'toss_winner']], drop_first=True)
y = data['winner']

# Check shapes of X and y
print(f'Features shape: {X.shape}, Target shape: {y.shape}')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))
print('Classification Report:')
print(classification_report(y_test, y_pred_rf, zero_division=1))

# Calculate win percentage for each team
team_wins = data['winner'].value_counts()
team_matches = data['team1'].value_counts() + data['team2'].value_counts()
win_percentage = (team_wins / team_matches).sort_values(ascending=False)

# Plot win percentage for each team
plt.figure(figsize=(8,5))
sns.barplot(x=win_percentage.index, y=win_percentage.values)
plt.xticks(rotation=90)
plt.xlabel('Team Name')
plt.ylabel('Winning Percentage')
plt.title('Winning Percentage of Each Team')
plt.show()
