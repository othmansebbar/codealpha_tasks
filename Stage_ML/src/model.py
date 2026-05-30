"""
Module de construction du CNN pour la reconnaissance de chiffres manuscrits.
"""

from tensorflow.keras import layers, models


def build_cnn(input_shape=(28, 28, 1), num_classes=10, name='Digit_CNN'):
    """
    Construit un CNN optimisé pour MNIST.
    """
    print("\nConstruction du modèle CNN...")
    
    model = models.Sequential(name=name)
    
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, name='Conv2D_1'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_1'))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='Conv2D_2'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_2'))
    model.add(layers.Dropout(0.25, name='Dropout_1'))
    
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dense(128, activation='relu', name='Dense_1'))
    model.add(layers.Dropout(0.5, name='Dropout_2'))
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))
    
    print(f"  Nom: {name}")
    print(f"  Paramètres totaux: {model.count_params():,}")
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec l'optimizer Adam.
    """
    from tensorflow.keras import optimizers
    
    print(f"\nCompilation du modèle (lr={learning_rate})...")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("  Optimizer: Adam")
    print("  Loss: categorical_crossentropy")
    print("  Metrics: accuracy")
    
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
    compile_model(model)
    print("\n✅ Module model testé avec succès!")