import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score, mean_absolute_error

def convert_typedejour_to_number(day):
    days_mapping = {
        'Sunday': 1,
        'Monday': 2,
        'Tuesday': 3,
        'Wednesday': 4,
        'Thursday': 5,
        'Friday': 6,
        'Saturday': 7
    }
    day_upper = day.capitalize()
    return days_mapping.get(day_upper)


data = pd.read_csv("/home/wahid/Electricity_Consumption_Prediction_System/main/Electricity_consumption_history.csv")


# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Sort data by date
data = data.sort_values(by='Date')

# Extract date features
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# remove the first column 'Date'
data = data.iloc[:,1:]

# Split the data into features "X" and target "Y"
X = data.iloc[:,[0,2,3,4,5]]
Y = data['Energie']

#split the data into training data and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=100)

regressor = DecisionTreeRegressor(criterion='squared_error', random_state=100)
regressor.fit(x_train, y_train)
# Make predictions on the test set
y_pred = regressor.predict(x_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2Score = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'r2 score: {r2Score}')

plot_tree(regressor)
plt.show()

plt.scatter(y_test.index,y_test,label='y actual',c='blue')
plt.scatter(y_test.index,y_pred,label='y_predicted',c='red')
plt.xlabel('index')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

RFregressor = RandomForestRegressor(n_estimators=100, random_state=100)
RFregressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = RFregressor.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2Score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 score: {r2Score}')
# Plot a tree from the forest (optional)
# You can't directly plot all trees in a Random Forest, but you can plot a single decision tree from it
plot_tree(RFregressor.estimators_[50]) # Choose a tree from the forest
plt.show()

plt.scatter(y_test.index, y_test, label='y actual', c='blue')
plt.scatter(y_test.index, y_pred, label='y_predicted', c='red')
plt.xlabel('index')
plt.ylabel('Y')
plt.title('Random Forest Regression')
plt.legend()
plt.show()
# User input for prediction
print("Enter the features for prediction:")
day_of_week = input("Enter day of week(Sunday/Monday/Tuesday/Wednesday/Thursday/Friday/Saturday): ")
tmax_input = int(input("Tmax: "))
day_input = int(input("Day: "))
month_input = int(input("Month: "))
year_input = int(input("Year: "))

# Prepare input for prediction
day_of_week = convert_typedejour_to_number(day_of_week)
input_data = pd.DataFrame({
    'Typedejour': [day_of_week],
    'Tmax': [tmax_input ],
    'Day': [day_input ],
    'Month': [month_input ],
    'Year': [year_input ]
})

prediction = regressor.predict(input_data)
print(f'Predicted Energy Consumption usig Decision Tree Regressor: {prediction[0]}')
Prediction = RFregressor.predict(input_data)
print(f'Predicted Energy Consumption using Random Forest Regressor: {prediction[0]}')
