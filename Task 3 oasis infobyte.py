import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv(
    r"C:\Users\vishal yadav\Downloads\spam.csv",
    encoding="latin-1"
)

# Select required columns
data = data[['v1', 'v2']]
data.columns = ['Category', 'Message']

# Bar chart
data['Category'].value_counts().plot(kind='bar')
plt.title("Spam and Ham Emails")
plt.show()

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(data['Message'])

# Target column
y = data['Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Test email
email = ["Congratulations! You won a free gift."]

email = cv.transform(email)
result = model.predict(email)

print("Prediction:", result[0])