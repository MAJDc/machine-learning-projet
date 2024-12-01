import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import math
import cvzone
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

# Charger le modèle YOLO
yolo_model = YOLO("Weights/best.pt")

# Définir les étiquettes des classes
class_labels = ['Avec Casque', 'Sans Casque']

# Charger l'image
image_path = "Media/riders_1.jpg"
img = cv2.imread(image_path)

# Effectuer la détection d'objets
results = yolo_model(img)

# Extraire les données pertinentes pour l'analyse
data = []  # Stocker les données de détection
predicted_labels = []  # Stocker les étiquettes prédites
true_labels = ['Avec Casque', 'Avec Casque', 'Avec Casque', 'Avec Casque', 'Sans Casque']  # Étiquettes réelles (ground truth)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = math.ceil((box.conf[0] * 100)) / 100  # Confiance
        cls = int(box.cls[0])  # Classe
        label = class_labels[cls]
        predicted_labels.append(label)
        data.append({'Classe': label, 'Confiance': conf})

# Créer un DataFrame
df = pd.DataFrame(data)

# Afficher le DataFrame (optionnel)
print(df)

# Visualisation avec Seaborn

# 1. Diagramme en barres pour le nombre de détections par classe
plt.figure(figsize=(8, 6))  # Ajuster la taille de la figure (optionnel)
sns.countplot(data=df, x='Classe', hue='Classe', palette='Set2', dodge=False)
plt.title("Nombre de Détections par Classe")
plt.xlabel("Classe")
plt.ylabel("Nombre")
plt.show()

# 2. Diagramme en boîte pour la distribution des scores de confiance par classe
plt.figure(figsize=(8, 6))  # Ajuster la taille de la figure (optionnel)
sns.boxplot(data=df, x='Classe', y='Confiance', palette='Set3')
plt.title("Distribution de la Confiance par Classe")
plt.xlabel("Classe")
plt.ylabel("Confiance")
plt.show()

# Matrice de confusion
if len(true_labels) == len(predicted_labels):  # Vérifier la correspondance des longueurs
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.show()
else:
    print("Erreur : Le nombre d'étiquettes réelles ne correspond pas au nombre d'étiquettes prédites.")

# Validation croisée à 4 plis (K-Fold CV)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
accuracies = []

# Simuler des données divisées en plis pour évaluer les performances
for train_index, test_index in kf.split(true_labels):
    train_true = [true_labels[i] for i in train_index]
    train_pred = [predicted_labels[i] for i in train_index]
    test_true = [true_labels[i] for i in test_index]
    test_pred = [predicted_labels[i] for i in test_index]

    # Évaluer la performance sur le jeu de test
    if len(test_true) == len(test_pred):  # Vérifier la correspondance des longueurs
        cm_fold = confusion_matrix(test_true, test_pred, labels=class_labels)
        accuracy = cm_fold.trace() / cm_fold.sum()
        accuracies.append(accuracy)
    else:
        print("Mismatch entre les étiquettes test réelles et prédites.")

# Résultats de la validation croisée
print(f"Précisions de la Validation Croisée : {accuracies}")
print(f"Précision Moyenne : {sum(accuracies) / len(accuracies):.2f}")
