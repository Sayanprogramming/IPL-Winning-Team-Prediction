1.Importing Libraries:

pandas for handling datasets.

matplotlib and seaborn for plotting and visualization.

train_test_split from sklearn.model_selection to split the data into training and testing sets.

StandardScaler from sklearn.preprocessing to scale the feature data.

RandomForestClassifier from sklearn.ensemble for building the classification model.

accuracy_score, confusion_matrix, and classification_report from sklearn.metrics to evaluate the model.

2.Loading the Dataset:

data = pd.read_csv('ipl match data.csv')

(The dataset is loaded from a CSV file into a DataFrame using pd.read_csv.)

3. Data Inspection:

print(data.head(10))

(This prints the first 10 rows of the dataset to inspect its structure and contents)

print("Null values in each column:")

print(data.isnull().sum())

(This prints the number of missing (null) values in each column of the dataset)

(This step helps in understanding if there are any missing values that need to be handled before moving forward)

4. Handling Missing Values:
   
data.dropna(subset=['winner'], inplace=True)

This drops any rows where the 'winner' column contains missing values.

(It’s important to remove these rows because the target variable ('winner') is crucial for the predictive modeling process.)

5. Feature Selection and Target Variable:

X = pd.get_dummies(data[['team1', 'team2', 'venue', 'toss_winner']], drop_first=True

y = data['winner']

X (features): The code selects four columns: 'team1', 'team2', 'venue', and 'toss_winner'. 

(These are the features that will be used to predict the match winner.Since these are categorical variables, pd.get_dummies is used to convert them into a numerical format using one-hot encoding.)

drop_first=True ensures that one of the categories for each column is dropped to avoid multicollinearity.

y (target): The target variable is 'winner', which represents the winning team.

6. Checking Shapes of Features and Target:
   
print(f'Features shape: {X.shape}, Target shape: {y.shape}')

(This prints the shapes of X (features) and y (target) to verify if they are correctly formatted.)

7. Splitting the Data:
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

The dataset is split into training and testing sets using an 80/20 split (test_size=0.2). 

The train_test_split function randomly splits the data while maintaining the order, with a fixed seed (random_state=42) for reproducibility.

8. Scaling the Features:
    
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

The features are scaled using StandardScaler to standardize the dataset. 

This ensures that each feature has a mean of 0 and a standard deviation of 1, which helps many machine learning algorithms perform better, especially those based on distance metrics (though Random Forests don’t strictly require scaling).

9. Checking Class Distribution:
    
print("Class distribution:")

print(y.value_counts())

This prints the number of occurrences of each class in the target variable 'winner'. 

It helps in understanding if there is any imbalance between classes, which could affect the model's performance.

10. Training the Random Forest Classifier:
    
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

A Random Forest classifier is initialized with 100 decision trees (n_estimators=100). 

(It is then trained using the training data (X_train and y_train). Random Forests are robust, non-linear classifiers that work well on structured data by averaging the results of multiple decision trees to improve performance.)

11. Making Predictions:
    
y_pred_rf = rf_model.predict(X_test)

After training, the model is used to predict the target variable for the test data (X_test).

12. Evaluating the Model:

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')

print('Confusion Matrix:')

print(confusion_matrix(y_test, y_pred_rf))

print('Classification Report:')

print(classification_report(y_test, y_pred_rf, zero_division=1))

Random Forest Accuracy: 52.75%

Confusion Matrix:

[[20  0  1  0  0  1  4  0  1  1  2  0  0  2  0  4  0  1]

 [ 0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 
 [ 2  0  1  0  0  1  0  0  0  0  1  0  0  1  0  0  0  1]
 
 [ 0  0  0  7  1  0  2  0  0  0  1  1  0  1  0  0  0  0]
 
 [ 0  0  0  0  2  0  0  0  0  0  2  0  0  0  0  1  0  0]
 
 [ 0  0  1  0  0  6  0  0  1  0  0  0  0  1  0  0  0  0]
 
 [ 2  0  1  1  0  0  6  0  1  0  1  0  0  1  0  3  0  1]
 
 [ 1  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0]
 
 [ 1  0  0  0  0  0  1  0 13  1  3  0  0  0  0  2  0  2]
 
 [ 0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  1  0]
 
 [ 3  0  0  0  0  0  2  0  1  0 17  0  0  2  0  0  0  1]
 
 [ 2  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0]
 
 [ 0  0  0  0  0  0  0  0  1  1  0  0  1  0  0  0  0  0]
 
 [ 2  0  0  0  0  0  3  0  3  0  1  0  0 14  0  2  0  0]
 
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  1]
 
 [ 1  0  0  2  0  0  0  0  3  0  4  0  1  0  0 14  0  1]
 
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]
 
 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  2  0  8]]
 
Classification Report:

                             precision    recall  f1-score   support

        Chennai Super Kings       0.56      0.54      0.55        37
            Deccan Chargers       1.00      1.00      1.00         2
             Delhi Capitals       0.25      0.14      0.18         7
           Delhi Daredevils       0.70      0.54      0.61        13
              Gujarat Lions       0.67      0.40      0.50         5
             Gujarat Titans       0.75      0.67      0.71         9
            Kings XI Punjab       0.33      0.35      0.34        17
       Kochi Tuskers Kerala       1.00      0.00      0.00         2
      Kolkata Knight Riders       0.52      0.57      0.54        23
       Lucknow Super Giants       0.40      0.67      0.50         3
             Mumbai Indians       0.52      0.65      0.58        26
              Pune Warriors       0.00      0.00      0.00         3
               Punjab Kings       0.50      0.33      0.40         3
           Rajasthan Royals       0.61      0.56      0.58        25
     Rising Pune Supergiant       1.00      0.50      0.67         4
Royal Challengers Bangalore       0.50      0.54      0.52        26
Royal Challengers Bengaluru       0.00      0.00      0.00         2
        Sunrisers Hyderabad       0.47      0.73      0.57        11


13. Calculating Win Percentage:

team_wins = data['winner'].value_counts()

team_matches = data['team1'].value_counts() + data['team2'].value_counts()

win_percentage = (team_wins / team_matches).sort_values(ascending=False)

This part of the code calculates the win percentage for each team:

team_wins: The number of matches won by each team.

team_matches: The total number of matches each team has played, which is the sum of the occurrences of a team being listed in 'team1' and 'team2'.

win_percentage: The win percentage for each team is computed by dividing the number of wins by the total number of matches and sorting the values in descending order.


14. Plotting Win Percentage:
     
plt.figure(figsize=(8,5))

sns.barplot(x=win_percentage.index, y=win_percentage.values)

plt.xticks(rotation=90)

plt.xlabel('Team Name')

plt.ylabel('Winning Percentage')

plt.title('Winning Percentage of Each Team')

plt.show()

![Screenshot 2024-10-16 201843](https://github.com/user-attachments/assets/10903fac-b9cd-46db-bc7d-887dcd8cd48c)

This section creates a bar plot using seaborn to visualize the winning percentages of each team:
The bar plot has teams on the x-axis and winning percentages on the y-axis.
