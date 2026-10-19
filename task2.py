import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load Dataset
df = pd.read_csv(r"C:\Users\USER\Downloads\task1\mynta_page(1-99).csv")

# Display first rows
print("Dataset Preview:")
print(df.head())

# Convert Price to numeric
df['Price'] = pd.to_numeric(
    df['Price'].astype(str).str.extract('(\d+)')[0],
    errors='coerce'
)

# Convert Discount to numeric
df['Discount'] = pd.to_numeric(
    df['Discount'].astype(str).str.extract('(\d+)')[0],
    errors='coerce'
)

# Remove rows with missing values
df.dropna(inplace=True)

# Create Classification Target
# High Discount = 1, Low Discount = 0
df['High_Discount'] = (df['Discount'] > 40).astype(int)

# Features
X = df[['Selling Price', 'Price']]

# Target Variable
y = df['High_Discount']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nRESULTS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))