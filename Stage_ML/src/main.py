"""
=============================================================================
SCRIPT PRINCIPAL - Handwritten Character Recognition
=============================================================================
Exécute le pipeline complet: chargement -> entraînement -> évaluation
=============================================================================

Usage:
    python src/main.py

Le modèle entraîné est sauvegardé dans models/
Les résultats sont sauvegardés dans results/
"""

import os
import sys

# Ajout du dossier src au path (pour les imports)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_preprocess_data
from model import build_cnn, compile_model
from train import train_model, plot_history
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    print_classification_report,
    visualize_predictions
)


def main():
    """
    Pipeline complet du projet HCR.
    """
    print("\n" + "="*60)
    print("HANDWRITTEN CHARACTER RECOGNITION - PIPELINE COMPLET")
    print("="*60)
    
    # --- Création des dossiers ---
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # --- 1. CHARGEMENT DES DONNÉES ---
       
    data = load_and_preprocess_data(test_size=0.1, random_state=42)
    
    # --- 2. CONSTRUCTION DU MODÈLE ---
    model = build_cnn(input_shape=(28, 28, 1), num_classes=10)
    model.summary()
    compile_model(model, learning_rate=0.001)
    
    # --- 3. ENTRAÎNEMENT ---
    history = train_model(
        model,
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val'],
        epochs=20,
        batch_size=128
    )
    
    # Visualisation de l'entraînement
    plot_history(history, save_path='results/training_history.png')
    
    # --- 4. ÉVALUATION ---
    results = evaluate_model(
        model,
        data['x_test'], data['y_test'],
        data['y_test_raw']
    )
    
    # Matrice de confusion
    plot_confusion_matrix(data['y_test_raw'], results['y_pred'])
    
    # Rapport de classification
    print_classification_report(data['y_test_raw'], results['y_pred'])
    
    # Visualisation des prédictions
    visualize_predictions(model, data['x_test'], data['y_test_raw'], num_samples=10)
    
    # --- 5. SAUVEGARDE FINALE ---
    model.save('models/final_model.keras')
    print("\n" + "="*60)
    print("PROJET TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print("Fichiers générés:")
    print("  ✅ models/best_model.keras    (meilleur modèle)")
    print("  ✅ models/final_model.keras    (modèle final)")
    print("  ✅ results/training_history.png")
    print("  ✅ results/confusion_matrix.png")
    print("  ✅ results/predictions_samples.png")
    print("  ✅ results/classification_report.txt")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    model, results = main()