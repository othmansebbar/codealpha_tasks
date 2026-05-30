"""
Module d'évaluation du modèle entraîné.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model, x_test, y_test_oh, y_test_raw):
    """
    Évalue le modèle sur le jeu de test.
    """
    print("\n" + "="*50)
    print("ÉVALUATION SUR LE JEU DE TEST")
    print("="*50)
    
    loss, accuracy = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"Loss:     {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print("\nGénération des prédictions...")
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc_sklearn = accuracy_score(y_test_raw, y_pred)
    print(f"Accuracy (sklearn): {acc_sklearn*100:.2f}%")
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Affiche et sauvegarde la matrice de confusion.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        cbar_kws={'label': 'Nombre d\'images'}
    )
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vérité terrain', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nMatrice de confusion sauvegardée: {save_path}")
    return cm


def print_classification_report(y_true, y_pred):
    """
    Affiche le rapport de classification détaillé.
    """
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=[f'Chiffre {i}' for i in range(10)],
        digits=4
    )
    print(report)
    
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    print("Rapport sauvegardé: results/classification_report.txt")


def visualize_predictions(model, x_test, y_test_raw, num_samples=10):
    """
    Visualise des prédictions aléatoires avec leurs probabilités.
    """
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    for i, idx in enumerate(indices):
        image = x_test[idx]
        true_label = y_test_raw[idx]
        
        pred_proba = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        pred_label = np.argmax(pred_proba)
        confidence = np.max(pred_proba) * 100
        
        axes[0, i].imshow(image.reshape(28, 28), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[0, i].set_title(
            f'Préd: {pred_label}\nVrai: {true_label}\nConf: {confidence:.1f}%',
            color=color, fontweight='bold', fontsize=9
        )
        axes[0, i].axis('off')
        
        axes[1, i].bar(range(10), pred_proba[0], color='steelblue', alpha=0.7)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title('Probabilités', fontsize=8)
        axes[1, i].axvline(x=pred_label, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Prédictions sur Images de Test', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/predictions_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPrédictions visualisées: results/predictions_samples.png")


if __name__ == "__main__":
    print("✅ Module evaluate chargé avec succès!")
    print("Utilisez evaluate_model() après l'entraînement.")