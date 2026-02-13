import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# Load Dataset
# =========================
data = pd.read_csv("customer_data.csv")

print("Dataset Loaded Successfully\n")
print(data.head())

# =========================
# Split Features & Target
# =========================
X = data.drop("Bought", axis=1)
y = data["Bought"]

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Feature Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Neural Network Model
# =========================
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1500, random_state=42)

print("\nTraining Model...")
model.fit(X_train, y_train)

# =========================
# Testing
# =========================
pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
acc = accuracy_score(y_test, pred)

print("\nModel Accuracy:", acc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# =========================
# New Customer Prediction
# =========================
sample = pd.DataFrame(
    [[30, 50000, 8, 3, 20]],
    columns=X.columns
)

sample_scaled = scaler.transform(sample)

result = model.predict(sample_scaled)

print("\nNew Customer Prediction:", result[0])