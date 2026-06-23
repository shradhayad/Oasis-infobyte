# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\vishal yadav\Downloads\Unemployment in India.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column into date format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Display first 5 rows
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Average unemployment rate by state
state_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

# Plot top 10 states with highest unemployment rate
top10 = state_unemployment.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top10.plot(kind='bar')
plt.title('Top 10 States by Unemployment Rate')
plt.xlabel('State')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Unemployment trend over time
monthly_unemployment = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(12,5))
plt.plot(monthly_unemployment.index, monthly_unemployment.values)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()

# Heatmap
pivot = df.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Region',
    columns='Date'
)

plt.figure(figsize=(12,8))
sns.heatmap(pivot, cmap='YlGnBu')
plt.title('State-wise Unemployment Heatmap')
plt.show()