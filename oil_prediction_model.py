# Import necessary libraries
import pandas as pd
import statsmodels.api as sm

# Data from the image
data = {
    'Oil_Gal': [
        275.30, 363.80, 164.30, 40.80, 94.30,
        230.90, 366.70, 300.60, 237.80, 121.40,
        31.40, 203.50, 441.10, 323.00, 52.50
    ],
    'Temp': [
        40, 27, 40, 73, 64,
        34, 9, 8, 23, 63,
        65, 41, 21, 38, 58
    ],
    'Insulation': [
        3, 3, 10, 6, 6,
        6, 6, 10, 10, 3,
        10, 6, 3, 3, 10
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set up independent variables (Temperature and Insulation) and dependent variable (Oil_Gal)
X = df[['Temp', 'Insulation']]
X = sm.add_constant(X)  # Add a constant term for the intercept
y = df['Oil_Gal']

# Build the linear regression model
model = sm.OLS(y, X).fit()

# Display model summary
print(model.summary())

# Predict oil consumption when Temp = 15°F and Insulation = 10 inches
new_data = pd.DataFrame({'const': [1], 'Temp': [15], 'Insulation': [10]})
predicted_oil = model.predict(new_data)

print(f"\nPredicted heating oil consumption (gallons) for Temp = 15°F and Insulation = 10 inches: {predicted_oil.iloc[0]:.2f} gallons")
