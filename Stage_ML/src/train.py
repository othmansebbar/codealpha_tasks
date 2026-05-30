"""
Module d'entraînement du modèle avec callbacks optimisés.
"""

from tensorflow.keras import callbacks, optimizers
import time


def get_callbacks(model_dir='models'):
    """
    Crée les callbacks pour un entraînement optimisé.
    
    Args:
        model_dir: Dossier de sauvegarde des modèles
    
    Returns:
        list: Callbacks Keras
    """
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    cb = [
        # Arrête si la validation stagne (évite l'overfitting)
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Réduit le learning rate si la loss stagne
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Sauvegarde le meilleur modèle automatiquement
        callbacks.ModelCheckpoint(
            filepath=f'{model_dir}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return cb


def train_model(model, x_train, y_train, x_val, y_val,
                epochs=20, batch_size=128, model_dir='models'):
    """
    Entraîne le modèle avec suivi automatique.
    
    Args:
        model: Modèle Keras compilé
        x_train, y_train: Données d'entraînement
        x_val, y_val: Données de validation
        epochs: Nombre d'epochs maximum
        batch_size: Taille des batches
        model_dir: Dossier de sauvegarde
    
    Returns:
        History: Historique d'entraînement
    """
    print("\n" + "="*50)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*50)
    print(f"  Epochs max: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train: {x_train.shape[0]} images")
    print(f"  Val:   {x_val.shape[0]} images")
    
    # Récupération des callbacks
    cb = get_callbacks(model_dir)
    
    # Chronométrage
    start_time = time.time()
    
    # Entraînement
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=cb,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    print(f"\nEntraînement terminé en {elapsed:.1f} secondes")
    print(f"Epochs réalisées: {len(history.history['accuracy'])}/{epochs}")
    
    return history


def plot_history(history, save_path='results/training_history.png'):
    """
    Affiche et sauvegarde les courbes d'entraînement.
    
    Args:
        history: Historique retourné par model.fit()
        save_path: Chemin de sauvegarde de la figure
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], 'b-', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
    axes[0].set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], 'b-', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation')
    axes[1].set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (Crossentropy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nFigure sauvegardée: {save_path}")


if __name__ == "__main__":
    print("✅ Module train chargé avec succès!")
    print("Utilisez train_model() pour entraîner votre modèle.")