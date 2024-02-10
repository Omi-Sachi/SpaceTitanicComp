

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the training data
training_data = pd.read_csv('train.csv')

# Remove rows with missing values
training_data['Cabin'].fillna('0/0/0', inplace=True)
Null_in_training_data = [training_data[col].isnull().sum() for col in training_data.columns]
training_data.fillna(0, inplace=True)

# Categorical data to perform one hot encoding

Unique_val = [training_data[col].nunique() for col in training_data.columns]
print(Unique_val)
print(training_data.head())


Null_in_Cabin = [training_data['Cabin'].isnull().sum()]
print(Null_in_Cabin)


# I want to use Cabin as it might be useful, lets clean it
training_data['Deck'] = training_data['Cabin'].apply(lambda x: x.split('/')[0])
training_data['Num'] = training_data['Cabin'].apply(lambda x: x.split('/')[1])
training_data['Side'] = training_data['Cabin'].apply(lambda x: x.split('/')[2])


training_data['Num'] = pd.to_numeric(training_data['Num'], errors='coerce')
training_data['Side'] = training_data['Side'].apply(lambda x: 0 if x == 'P' else (1 if x == 'S' else 2 if x == '0' else None))
# Create a mapping of deck letters to numerical values
deck_mapping = {
    '0' : 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,
    'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26
}

# Apply the mapping to create a new 'deck_ordinal' column
training_data['Deck'] = training_data['Deck'].map(deck_mapping)

# Categorical data to perform one hot encoding
features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Num', 'Side' ]

# Display a subset of the DataFrame to check the changes
print(training_data[features].isnull().sum())

X = pd.get_dummies(training_data[features]).astype(int)

print(training_data.columns)


# Target variable
y = training_data['Transported'].astype(int)

# Creating the model
rt_model = RandomForestRegressor()
rt_model.fit(X, y)

# Load the testing data
testing_data = pd.read_csv('test.csv')

# do what i did for deck in test.
testing_data['Cabin'].fillna('0/0/0', inplace=True)

# Remove rows with missing values
Null_in_testing_data = [testing_data[col].isnull().sum() for col in testing_data.columns]
testing_data.fillna(0, inplace=True)


# Convert the column to integer dtype
Null_in_Cabin_test = [testing_data['Cabin'].isnull().sum()]
print(Null_in_Cabin_test)
print(testing_data["Cabin"].describe())

# I want to use Cabin as it might be useful, lets clean it
testing_data['Deck'] = testing_data['Cabin'].apply(lambda x: x.split('/')[0])
testing_data['Num'] = testing_data['Cabin'].apply(lambda x: x.split('/')[1])
testing_data['Side'] = testing_data['Cabin'].apply(lambda x: x.split('/')[2])


testing_data['Num'] = pd.to_numeric(testing_data['Num'], errors='coerce')
testing_data['Side'] = testing_data['Side'].apply(lambda x: 0 if x == 'P' else (1 if x == 'S' else 2 if x == '0' else None))

# Apply the mapping to create a new 'deck_ordinal' column
testing_data['Deck'] = testing_data['Deck'].map(deck_mapping)
# Perform one hot encoding on testing data
x_test = pd.get_dummies(testing_data[features]).astype(int)

# Make predictions on the testing data, round them down
predictions = rt_model.predict(x_test)
predictions = predictions.round().astype(int)

# If my prediction is greater then 0.5 i will turn it to false
predictions = predictions > 0.5

# Create a DataFrame for submission
output = pd.DataFrame({'PassengerId': testing_data['PassengerId'], 'Transported': predictions})

# Save the submission file
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
