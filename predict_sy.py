##Predicting Mechanical Yield Strength of Alloys Using Statistical and Machine Learning Techniques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import warnings

#Loading Data 
df = pd.read_csv(r"D:/Downloads/Data.csv")
print(df.head())

#Data Cleaning
df = df.select_dtypes(include=[np.number])  # Keep numeric columns only
imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)

#Feature Selection
X = df.drop(columns=['Sy'], errors='ignore')  # Independent variables
y = df['Sy'] if 'Sy' in df.columns else df.iloc[:, -1]  # Target variable

#Correlation Heatmap for better understanding of features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Regression Attempts
results = []
attempts = {
    "Attempt 1: Single Feature (Bhn)": ['Bhn'],
    "Attempt 2: Two Features (Bhn + A5)": ['Bhn', 'A5'],
    "Attempt 3: Three Features (Bhn + A5 + E)": ['Bhn', 'A5', 'E'],
    "Attempt 4: All Features": list(X.columns)
}
plt.figure(figsize=(12, 8))
for i, (label, cols) in enumerate(attempts.items(), 1):
    model = LinearRegression()
    model.fit(X_train[cols], y_train)
    y_pred = model.predict(X_test[cols])
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    results.append((label, r2, rmse))

    #Plot for all checked attempts
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'm--')
    plt.xlabel("Actual Sy")
    plt.ylabel("Predicted Sy")
    plt.title(f"{label}\nR²={r2:.3f}, RMSE={rmse:.2f}")
plt.tight_layout()
plt.show()

#Results
results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE"])
print("Model Comparison:")
print(results_df.sort_values(by="R2", ascending=False))

#Final Model Selection
best_label = results_df.sort_values(by="R2", ascending=False).iloc[0]['Model']
best_features = attempts[best_label]
final_model = LinearRegression()
final_model.fit(X_train[best_features], y_train)
final_preds = final_model.predict(X_test[best_features])

#Final Plot
plt.figure(figsize=(6, 5))
plt.scatter(y_test, final_preds, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--')
plt.xlabel("Actual Sy")
plt.ylabel("Predicted Sy")
plt.title(f"Final Model: {best_label}")
plt.grid(True)
plt.show()

#User Input Prediction
#Example:    
# Ultimate tensile strength in MPa (e.g., 450 MPa)
# Enter A5: 32.5       # Percent elongation at break (e.g., 32.5%)
# Enter Bhn: 145       # Brinell Hardness Number (e.g., 145)
# Enter E: 207000      # Modulus of Elasticity in MPa (e.g., 207,000 MPa)
# Enter G: 79000       # Shear Modulus in MPa (e.g., 79,000 MPa)
# Enter mu: 0.3        # Poisson's ratio (e.g., 0.3 typical for metals)
# Enter Ro: 7860       # Density in kg/m³ (e.g., 7860 kg/m³ for steel)
# Enter pH: 7.0        # pH value (neutral environment assumed)
# Enter HV: 160        # Vickers Hardness (e.g., 160)

print("Enter your own values to predict Yield Strength (Sy):")
user_values = []
for feature in best_features:
    val = float(input(f"Enter {feature}: "))
    user_values.append(val)

user_array = np.array(user_values).reshape(1, -1)
pred_sy = final_model.predict(user_array)[0]
print(f"Predicted Yield Strength (Sy): {pred_sy:.2f} MPa")
