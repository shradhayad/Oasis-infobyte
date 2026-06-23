import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:\Users\vishal yadav\Downloads\Advertising.csv")

# Visualization
plt.figure(figsize=(6,4))
plt.scatter(data["TV"], data["Sales"])
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.show()

# Features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Accuracy
print("Model Score:", model.score(X_test, y_test))

# Sample predictions
print("\nPredicted Sales:")
print(predictions[:5])