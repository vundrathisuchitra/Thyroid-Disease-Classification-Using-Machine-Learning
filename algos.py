#!/usr/bin/env python
# coding: utf-8

# ==============================
# IMPORT LIBRARIES
# ==============================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 224
DATASET_PATH = "dataset/train"   # classes inside this folder


# ==============================
# LOAD IMAGE DATASET
# ==============================
def load_dataset(dataset_path):
    X, y = [], []
    class_names = sorted(os.listdir(dataset_path))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)

            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image / 255.0
                X.append(image.flatten())
                y.append(label)

    return np.array(X), np.array(y), class_names


print("[INFO] Loading dataset...")
X, y, class_names = load_dataset(DATASET_PATH)

print("Classes:", class_names)
print("Total samples:", X.shape[0])


# ==============================
# TRAIN–TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# ML MODELS
# ==============================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}


# ==============================
# TRAIN & EVALUATE MODELS
# ==============================
for name, model in models.items():
    print(f"\n[INFO] Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = (acc, prec, rec, f1)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))


# ==============================
# ACCURACY COMPARISON GRAPH
# ==============================
model_names = list(results.keys())
accuracy = [results[m][0] for m in model_names]

plt.figure(figsize=(8,5))
plt.bar(model_names, accuracy)
plt.title("Accuracy Comparison of ML Models")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()


# ==============================
# PRECISION / RECALL / F1 GRAPH
# ==============================
precision = [results[m][1] for m in model_names]
recall = [results[m][2] for m in model_names]
f1 = [results[m][3] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-Score')

plt.xticks(x, model_names)
plt.ylabel("Score")
plt.title("Precision, Recall & F1-Score Comparison")
plt.legend()
plt.show()


# ==============================
# CONFUSION MATRIX (BEST MODEL)
# ==============================
best_model_name = max(results, key=lambda k: results[k][0])
best_model = models[best_model_name]

print(f"\n[INFO] Best Model: {best_model_name}")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.colorbar()

plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ==============================
# FINAL ACCURACY OUTPUT
# ==============================
print("\nFINAL ACCURACY RESULTS")
for model, score in results.items():
    print(f"{model}: {score[0]*100:.2f}%")

import joblib

joblib.dump(models["Logistic Regression"], "lr_model.pkl")
joblib.dump(scaler, "scaler.pkl")