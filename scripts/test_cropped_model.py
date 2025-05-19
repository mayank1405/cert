import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load test data
X_test = np.load("cropped_dataset/X_test.npy")
y_test = np.load("cropped_dataset/y_test.npy")

# Load the trained model
model = tf.keras.models.load_model("models/certificate_cropped_deepcnn.h5")

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n🧪 Test Accuracy: {acc * 100:.2f}%")

# Predict and analyze
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification report
print("\n📊 Classification Report (Test Set):\n")
print(classification_report(y_test, y_pred, target_names=["Authentic", "Tampered"]))
