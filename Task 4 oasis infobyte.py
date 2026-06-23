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

# Keep only useful columns
data = data[['v1', 'v2']]
data.columns = ['Category', 'Message']

# Visualization
data['Category'].value_counts().plot(kind='bar')
plt.title("Spam vs Ham Emails")
plt.show()

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])

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