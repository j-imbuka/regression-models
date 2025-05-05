
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data from the table
lengths = np.array([3, 4, 5, 6, 7])
made = np.array([84, 88, 61, 61, 44])
missed = np.array([17, 31, 47, 64, 90])
total = made + missed
success_rate = made / total

# Create a DataFrame
df = pd.DataFrame({
    'Length': lengths,
    'Made': made,
    'Missed': missed,
    'Total': total,
    'SuccessRate': success_rate
})

# Prepare data for logistic regression
X = sm.add_constant(df['Length'])  # add constant for intercept
y = df['Made'] / df['Total']       # proportion of successes
weights = df['Total']              # weights = total attempts

# Fit logistic regression model using GLM with binomial family
model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
result = model.fit()

# Predict probabilities over a range of lengths
x_pred = np.linspace(3, 7, 100)
X_pred = sm.add_constant(x_pred)
y_pred = result.predict(X_pred)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(df['Length'], df['SuccessRate'], color='blue', label='Observed Success Rate')
plt.plot(x_pred, y_pred, color='red', label='Logistic Regression Prediction')
plt.xlabel('Putt Length (feet)')
plt.ylabel('Proportion of Putts Made')
plt.title('Logistic Regression Model for Golf Putts')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the model summary
print(result.summary())
