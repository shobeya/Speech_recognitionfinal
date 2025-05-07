import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the features DataFrame
df = pd.read_pickle("ravdess_mfcc_features.pkl")

# Make sure required columns exist
assert 'mfcc' in df.columns and 'emotion' in df.columns, "Missing 'mfcc' or 'emotion' columns."

# Extract features (X) and labels (y)
X = np.array(df['mfcc'].tolist())
y = df['emotion']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Select model – Logistic Regression (optimized)
# model = LogisticRegression(max_iter=1000, random_state=42)

# Or try Random Forest (recommended)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# 1. Display predicted emotions first
print("Predicted Emotions:")
for i in range(len(y_pred)):
    print(f"Sample {i+1}: Predicted Emotion - {y_pred[i]}")

# 2. Now display the evaluation metrics
print("\nEvaluation Metrics:")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy}")

# Generate classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Add Single Data Visualization – Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
