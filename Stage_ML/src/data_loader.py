"""
Module de chargement et prétraitement des données MNIST.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_raw_data():
    """
    Charge le dataset MNIST depuis Keras.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    print("Chargement du dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"  Train: {x_train.shape[0]} images")
    print(f"  Test:  {x_test.shape[0]} images")
    print(f"  Dimensions: {x_train.shape[1:]} pixels")
    
    return (x_train, y_train), (x_test, y_test)


def normalize_and_reshape(x_train, x_test):
    """
    Normalise les pixels [0,255] -> [0,1] et ajoute la dimension canal.
    
    Args:
        x_train: Images d'entraînement (N, 28, 28)
        x_test: Images de test (N, 28, 28)
    
    Returns:
        tuple: Images normalisées et reshapées (N, 28, 28, 1)
    """
    print("\nPrétraitement des images...")
    
    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape: ajout de la dimension canal (niveaux de gris = 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    print(f"  Valeurs normalisées: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"  Nouvelle shape: {x_train.shape}")
    
    return x_train, x_test


def encode_labels(y_train, y_test, num_classes=10):
    """
    Convertit les labels en encodage one-hot.
    
    Args:
        y_train: Labels d'entraînement
        y_test: Labels de test
        num_classes: Nombre de classes (10 pour MNIST)
    
    Returns:
        tuple: Labels encodés
    """
    print(f"\nEncodage one-hot des labels ({num_classes} classes)...")
    
    y_train_oh = to_categorical(y_train, num_classes)
    y_test_oh = to_categorical(y_test, num_classes)
    
    print(f"  Exemple: label {y_train[0]} -> {y_train_oh[0]}")
    
    return y_train_oh, y_test_oh


def split_validation(x_train, y_train_oh, test_size=0.1, random_state=42):
    """
    Sépare une partie du train pour la validation.
    
    Args:
        x_train: Images d'entraînement
        y_train_oh: Labels one-hot
        test_size: Proportion pour la validation (défaut: 10%)
        random_state: Graine pour la reproductibilité
    
    Returns:
        tuple: (x_train, x_val, y_train_oh, y_val_oh)
    """
    print(f"\nCréation du jeu de validation ({test_size*100:.0f}%)...")
    
    x_train, x_val, y_train_oh, y_val_oh = train_test_split(
        x_train, y_train_oh,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_oh  # Répartition équilibrée des classes
    )
    
    print(f"  Train final:   {x_train.shape[0]} images")
    print(f"  Validation:    {x_val.shape[0]} images")
    
    return x_train, x_val, y_train_oh, y_val_oh


def load_and_preprocess_data(test_size=0.1, random_state=42):
    """
    Pipeline complet: charge, normalise, encode et split les données.
    
    Args:
        test_size: Proportion pour la validation
        random_state: Graine pour reproductibilité
    
    Returns:
        dict: Tous les jeux de données organisés
    """
    print("\n" + "="*50)
    print("PIPELINE DE PRÉTRAITEMENT")
    print("="*50)
    
    # 1. Chargement
    (x_train, y_train), (x_test, y_test) = load_raw_data()
    
    # 2. Normalisation et reshape
    x_train, x_test = normalize_and_reshape(x_train, x_test)
    
    # 3. Encodage des labels
    y_train_oh, y_test_oh = encode_labels(y_train, y_test)
    
    # 4. Split validation
    x_train, x_val, y_train_oh, y_val_oh = split_validation(
        x_train, y_train_oh, test_size, random_state
    )
    
    print("\n" + "="*50)
    print("RÉPARTITION FINALE")
    print("="*50)
    print(f"  Entraînement:  {x_train.shape[0]} images")
    print(f"  Validation:    {x_val.shape[0]} images")
    print(f"  Test:          {x_test.shape[0]} images")
    print("="*50)
    
    # ✅ Dictionnaire pour accès facile dans main.py
    return {
        'x_train': x_train,
        'y_train': y_train_oh,
        'x_val': x_val,
        'y_val': y_val_oh,
        'x_test': x_test,
        'y_test': y_test_oh,
        'y_test_raw': y_test  # Labels originaux (0-9) pour l'évaluation
    }


if __name__ == "__main__":
    # Test du module
    data = load_and_preprocess_data()
    print("\n✅ Module data_loader testé avec succès!")
    print(f"Clés disponibles: {list(data.keys())}")