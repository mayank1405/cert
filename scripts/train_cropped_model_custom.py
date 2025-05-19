import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed data
X_train = np.load("cropped_dataset/X_train.npy")
X_val = np.load("cropped_dataset/X_val.npy")
y_train = np.load("cropped_dataset/y_train.npy")
y_val = np.load("cropped_dataset/y_val.npy")

# Define a custom CNN model
model = models.Sequential([
layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), padding='same'),  
    layers.BatchNormalization(),  
    layers.Activation('relu'),  
    layers.MaxPooling2D(2, 2),  
    
    layers.Conv2D(64, (3, 3), padding='same'),  
    layers.BatchNormalization(),  
    layers.Activation('relu'),  
    layers.MaxPooling2D(2, 2),  
    
    layers.Conv2D(128, (3, 3), padding='same'),  
    layers.BatchNormalization(),  
    layers.Activation('relu'),  
    layers.MaxPooling2D(2, 2),  
    
    layers.Conv2D(256, (3, 3), padding='same'),  
    layers.BatchNormalization(),  
    layers.Activation('relu'),  
    layers.MaxPooling2D(2, 2),  
    
    layers.Flatten(),  
    layers.Dense(256, activation='relu'),  
    layers.Dropout(0.5),  
    layers.Dense(1, activation='sigmoid')  

])

# Compile the model
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(patience=7, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
X_train, y_train,
epochs=50,
batch_size=16,
validation_data=(X_val, y_val),
callbacks=[early_stop]
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/certificate_cropped_deepcnn.h5")

# Evaluate the model
loss, acc = model.evaluate(X_val, y_val)
print(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")

# Predict and show confusion matrix
y_pred = (model.predict(X_val) > 0.5).astype("int32")
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=["Authentic", "Tampered"]))