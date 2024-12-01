from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Charger les données d'entraînement (caractéristiques et étiquettes)
X = np.load("train_features.npy")  # Caractéristiques HOG extraites
y = np.load("train_labels.npy")    # Étiquettes (0: Without Helmet, 1: With Helmet)

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, "helmet_svm_model.pkl")
