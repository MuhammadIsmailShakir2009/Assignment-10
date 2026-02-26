"""utils/model_builder.py — Build CNN with Keras."""

from tensorflow import keras
from tensorflow.keras import layers


def build(img_size: int, n_classes: int, lr: float = 0.001) -> keras.Model:
    """Build and compile a CNN for image classification."""
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="NeuralForge_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summary_str(model: keras.Model) -> str:
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
