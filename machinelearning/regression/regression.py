from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample data
data = {'Day': [1, 2, 3, 4, 5, 6],
        'Temp': [31, 32, 33, 34, 35, 36]}
df = pd.DataFrame(data)

# Prepare features and labels
X = df[['Day']]
y = df['Temp']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make prediction
d = 8
predicted = model.predict([[d]])
print(f"\nPredicted Temp on Day {d} is: {predicted}")
